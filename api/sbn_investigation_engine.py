"""
SBN InvestigationLoopAgent — DyTopo-inspired engine
Pure Python, no LangGraph imports.

Design: InvestigationLoopAgent_DyTopo_Design.md
        InvestigationLoopAgent_Implementation.md

Topology mapping:
  DyTopo concept          → SBN equivalent
  Worker agents (A_i)     → Tools: FlowLogTool, KafkaOffsetTool,
                             RoutingDecisionTool, OrgConfigTool
  Query descriptor s_q    → investigator_query  (one sentence: what is needed)
  Key descriptor s_k      → TOOL_KEYS[name]     (static: what the tool provides)
  Adjacency matrix A^(t)  → activated_tools     (rebuilt each round via cosine)
  Manager halting y^(t)   → halt flag + halt_reason (deterministic rules)
  Round goal C_task^(t)   → round_goal          (updated by manager each round)
  T_max                   → SBNInvestigationConfig.t_max  (default 5)
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import httpx
import numpy as np

# ── Sentence encoder (loaded once, cached) ────────────────────────────────────
# Uses the same all-MiniLM-L6-v2 model as dytopo_engine for local/offline use.
# In production the SBN stack calls text-embedding-3-small via SAP AI Core;
# set mock_mode=False and provide llm_base_url to use that path instead.
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def _encode_local(texts: list[str]) -> np.ndarray:
    """L2-normalised (N, 384) float32 embeddings via local MiniLM."""
    enc = _get_encoder()
    vecs = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype=np.float32)


# ── Tool key descriptors (§4 of design doc) ───────────────────────────────────
TOOL_KEYS: dict[str, str] = {
    "FlowLogTool": (
        "Provides the full step-by-step routing flow trace for a document, "
        "showing which routing steps executed, which step failed, and at what timestamp."
    ),
    "KafkaOffsetTool": (
        "Reports whether a document message was produced to Kafka, the current "
        "consumer offset, and whether the message was ever delivered downstream."
    ),
    "RoutingDecisionTool": (
        "Returns the routing decision record: the routing method used, the destination "
        "PostURL, the buyer orgId, and whether a valid routing rule was matched."
    ),
    "OrgConfigTool": (
        "Returns the buyer organisation's current OrgPreferences: the active PostURL, "
        "routing flags, feature toggles, and the last-modified timestamp of the configuration."
    ),
}

# Pre-computed at startup via precompute_tool_embeddings()
_TOOL_KEY_EMBEDDINGS: dict[str, np.ndarray] = {}


def precompute_tool_embeddings() -> None:
    """
    Embed all tool keys once at startup and cache in memory.
    Cost: 4 embedding calls, done once per pod lifetime (§6 of impl doc).
    Call this from your FastAPI lifespan handler.
    """
    keys = list(TOOL_KEYS.keys())
    texts = [TOOL_KEYS[k] for k in keys]
    vecs = _encode_local(texts)
    for name, vec in zip(keys, vecs):
        _TOOL_KEY_EMBEDDINGS[name] = vec  # already L2-normalised by _encode_local


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class SBNInvestigationConfig:
    tau_edge: float = 0.35              # cosine threshold for tool activation (§8)
    t_max: int = 5                      # hard round cap (§9 condition C)
    confidence_threshold: float = 0.90  # halt when investigator is this certain (§9 condition A)
    mock_mode: bool = True
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:8b"
    temperature: float = 0.0            # deterministic for audit reproducibility
    max_tokens: int = 1200
    java_invoice_service_base: str = "http://java-invoice-service/private/api/v1"


# ── Data types ────────────────────────────────────────────────────────────────
@dataclass
class MemoryEntry:
    round: int
    tool: str
    result: str
    summary: str


@dataclass
class RoundResult:
    round: int
    round_goal: str
    investigator_query: str
    investigator_summary: str
    confidence: float
    activated_tools: list[str]
    tool_scores: dict[str, float]
    tool_results: dict[str, str]
    halt: bool
    halt_reason: str          # "confidence" | "exhausted" | "cap" | "repeat" | ""
    root_cause: str | None
    suggested_fix: str | None
    ops_summary: str | None
    eng_trace: str | None

    def to_dict(self) -> dict:
        return {
            "round":                self.round,
            "round_goal":           self.round_goal,
            "investigator_query":   self.investigator_query,
            "investigator_summary": self.investigator_summary,
            "confidence":           self.confidence,
            "activated_tools":      self.activated_tools,
            "tool_scores":          self.tool_scores,
            "tool_results":         self.tool_results,
            "halt":                 self.halt,
            "halt_reason":          self.halt_reason,
            "root_cause":           self.root_cause,
            "suggested_fix":        self.suggested_fix,
            "ops_summary":          self.ops_summary,
            "eng_trace":            self.eng_trace,
        }


# ── Semantic matching (§8) ────────────────────────────────────────────────────
def match_tools(investigator_query: str, tau: float) -> tuple[list[str], dict[str, float]]:
    """
    Embed investigator_query, compute cosine similarity against cached tool key
    embeddings, return (activated_tools, tool_scores).
    Requires precompute_tool_embeddings() to have been called first.
    """
    if not _TOOL_KEY_EMBEDDINGS:
        precompute_tool_embeddings()

    q_vec = _encode_local([investigator_query])[0]   # (384,) L2-normalised

    scores: dict[str, float] = {
        name: float(np.dot(q_vec, key_emb))
        for name, key_emb in _TOOL_KEY_EMBEDDINGS.items()
    }
    activated = [name for name, score in scores.items() if score > tau]
    return activated, scores


# ── LLM call — investigator node (§7.1) ──────────────────────────────────────
_INVESTIGATOR_SYSTEM = """\
You are an investigation agent diagnosing a B2B document routing failure.

Given the current round goal and accumulated investigation memory, output a JSON object:
{
  "query":      "one sentence — what information you currently need to make progress",
  "summary":    "one sentence — what you learned from the last round's tool results (empty string on round 0)",
  "confidence": 0.0
}

Rules:
- query must be a specific, answerable question about the document or org config
- summary must state a concrete finding, not just 'I called a tool'
- confidence: 0.0=no idea, 0.5=strong hypothesis, 0.9+=confirmed root cause
- Respond ONLY with valid JSON, no markdown, no explanation"""

_SYNTHESISE_SYSTEM = """\
You are a root cause analyst for B2B document routing failures.

Given investigation round summaries and a halt reason, output a JSON object:
{
  "root_cause":    "one or two sentences — the definitive root cause",
  "suggested_fix": "one or two sentences — the actionable fix for the user",
  "ops_summary":   "2-4 plain-language sentences for an operations user. Include: what failed, why, what to do, the document ID.",
  "eng_trace":     "technical summary: round count, halt reason, tool activation per round with cosine scores, root cause chain of evidence"
}

Respond ONLY with valid JSON, no markdown, no explanation."""


async def _call_llm(system: str, user: str, cfg: SBNInvestigationConfig) -> dict:
    """POST to OpenAI-compatible endpoint; parse JSON from response."""
    payload = {
        "model": cfg.llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": cfg.temperature,
        "max_tokens":  cfg.max_tokens,
        "stream":      False,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{cfg.llm_base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

    # Extract JSON — same 3-tier fallback as dytopo_engine
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        blob = m.group()
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', blob)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
    return {"query": raw[:300], "summary": "", "confidence": 0.0,
            "root_cause": raw[:300], "suggested_fix": "", "ops_summary": raw[:300], "eng_trace": ""}


async def _investigator_llm(
    round_t: int,
    round_goal: str,
    memory: list[MemoryEntry],
    cfg: SBNInvestigationConfig,
) -> dict:
    memory_text = "\n".join(
        f"Round {e.round} [{e.tool}]: {e.summary}" for e in memory
    ) or "No prior rounds."

    user = (
        f"Round goal: {round_goal}\n\n"
        f"Investigation memory so far:\n{memory_text}\n\n"
        f"Respond with JSON."
    )
    return await _call_llm(_INVESTIGATOR_SYSTEM, user, cfg)


async def _synthesise_llm(
    payload_id: str,
    org_id: str,
    user_question: str,
    memory: list[MemoryEntry],
    halt_reason: str,
    confidence: float,
    tool_scores_per_round: list[dict[str, float]],
    cfg: SBNInvestigationConfig,
) -> dict:
    summaries = "\n".join(
        f"Round {e.round} [{e.tool}]: {e.summary}" for e in memory
    )
    activation_trace = "\n".join(
        f"  round {i}: " + ", ".join(f"{t} ({s:.2f})" for t, s in scores.items() if s > cfg.tau_edge)
        for i, scores in enumerate(tool_scores_per_round)
    )
    user = (
        f"Document: {payload_id}\n"
        f"Buyer org: {org_id}\n"
        f"User question: {user_question}\n\n"
        f"Investigation round summaries:\n{summaries}\n\n"
        f"Tool activation trace:\n{activation_trace}\n\n"
        f"Halt reason: {halt_reason}\n"
        f"Final confidence: {confidence}\n\n"
        f"Respond with JSON."
    )
    return await _call_llm(_SYNTHESISE_SYSTEM, user, cfg)


# ── Mock mode — scripted responses (no LLM) ───────────────────────────────────
# Mirrors the 3-round worked example from §10 of the design doc.
# payload_id="INV-78432", org_id="BNO-456001"

_MOCK_INVESTIGATOR: list[dict] = [
    {
        "query":      "What routing steps executed for INV-78432 and where did the process stop?",
        "summary":    "",
        "confidence": 0.10,
    },
    {
        "query":      "What routing method and destination PostURL was configured for buyer BNO-456001?",
        "summary":    "RoutingPublishStep failed at 09:14:32; Kafka message was produced but never consumed downstream.",
        "confidence": 0.45,
    },
    {
        "query":      "Is the buyer BNO-456001 PostURL currently blank and has it been blank since September?",
        "summary":    "RoutingDecisionTool shows destination_url=null; OrgConfigTool confirms PostURL is empty.",
        "confidence": 0.95,
    },
]

_MOCK_TOOLS: dict[str, dict[str, str]] = {
    # round → tool → result
    "0": {
        "FlowLogTool":    "Steps: DocReceived OK, SchemaValidation OK, RoutingLookup OK, RoutingPublishStep FAILED at 09:14:32.",
        "KafkaOffsetTool":"Kafka message produced: YES. Consumer offset: never consumed. Partition 3, offset 88441.",
    },
    "1": {
        "RoutingDecisionTool": "routing_method: DIRECT_POST. destination_url: null. buyer_org_id: BNO-456001. routing_rule_matched: false.",
        "OrgConfigTool":       "postURL: '' (empty). routing_enabled: true. last_modified: 2026-09-14T11:22:00Z.",
    },
    "2": {
        "OrgConfigTool":       "postURL: '' (confirmed still empty). last_modified: 2026-09-14T11:22:00Z.",
        "RoutingDecisionTool": "destination_url: null (confirmed).",
    },
}

_MOCK_SYNTHESIS = {
    "root_cause":    "Buyer BNO-456001 has no active PostURL configured in OrgPreferences (blank since 2026-09-14). RoutingPublishStep failed because destination_url resolved to null.",
    "suggested_fix": "Ask buyer admin to configure PostURL in OrgPreferences for BNO-456001, then resubmit INV-78432.",
    "ops_summary":   "Invoice INV-78432 failed routing. Reason: Buyer BNO-456001 has no active PostURL configured in OrgPreferences (blank since 2026-09-14). Fix: Ask buyer admin to configure PostURL in OrgPreferences, then resubmit.",
    "eng_trace":     "round_count: 3 | halt_reason: confidence (0.95) | tool_activation: round 0: FlowLogTool (0.82), KafkaOffsetTool (0.61) | round 1: RoutingDecisionTool (0.88), OrgConfigTool (0.71) | round 2: OrgConfigTool (0.89), RoutingDecisionTool (0.44) | root_cause: PostURL null in routing decision; OrgPreferences.postURL empty since 2026-09-14.",
}


async def _mock_investigator(round_t: int) -> dict:
    await asyncio.sleep(0.05)
    r = min(round_t, len(_MOCK_INVESTIGATOR) - 1)
    return dict(_MOCK_INVESTIGATOR[r])


async def _mock_tool_dispatch(round_t: int) -> dict[str, str]:
    await asyncio.sleep(0.05)
    r = min(round_t, 2)
    return dict(_MOCK_TOOLS.get(str(r), {}))


# ── Tool dispatch (real HTTP calls — §7.3) ────────────────────────────────────
async def _real_tool_dispatch(
    activated_tools: list[str],
    payload_id: str,
    org_id: str,
    cfg: SBNInvestigationConfig,
) -> dict[str, str]:
    """Call each activated tool in parallel against the Java Invoice Service."""
    async def _call(name: str) -> tuple[str, str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                if name == "OrgConfigTool":
                    url = f"{cfg.java_invoice_service_base}/org-preferences/{org_id}"
                elif name == "FlowLogTool":
                    url = f"{cfg.java_invoice_service_base}/flow-log/{payload_id}"
                elif name == "KafkaOffsetTool":
                    url = f"{cfg.java_invoice_service_base}/kafka-offset/{payload_id}"
                elif name == "RoutingDecisionTool":
                    url = f"{cfg.java_invoice_service_base}/routing-decision/{payload_id}"
                else:
                    return name, f"ERROR: unknown tool {name}"
                resp = await client.get(url)
                resp.raise_for_status()
                return name, resp.text
        except Exception as e:
            return name, f"ERROR: {e}"

    pairs = await asyncio.gather(*[_call(name) for name in activated_tools])
    return dict(pairs)


# ── Manager — deterministic halt rules (§9 / §7.4) ───────────────────────────
def _evaluate_manager(
    round_t: int,
    confidence: float,
    activated_tools: list[str],
    tool_results: dict[str, str],
    prev_activated_tools: list[str],
    prev_tool_results: dict[str, str],
    investigator_summary: str,
    cfg: SBNInvestigationConfig,
) -> tuple[bool, str, str]:
    """
    Returns (halt, halt_reason, next_round_goal).
    Halt conditions evaluated in priority order per §9:
      A: confidence >= threshold
      B: no tools activated (information exhausted)
      C: round >= T_max (hard cap)
      D: same tools + same results (no progress)
    """
    # Condition A — confidence threshold
    if confidence >= cfg.confidence_threshold:
        return True, "confidence", ""

    # Condition B — information exhausted
    if not activated_tools:
        return True, "exhausted", ""

    # Condition C — hard cap
    if round_t >= cfg.t_max:
        return True, "cap", ""

    # Condition D — no progress
    if (
        sorted(activated_tools) == sorted(prev_activated_tools)
        and tool_results == prev_tool_results
    ):
        return True, "repeat", ""

    next_goal = (
        f"Based on finding '{investigator_summary}', "
        f"now investigate the next unresolved aspect of the routing failure."
    )
    return False, "", next_goal


# ── Main loop ─────────────────────────────────────────────────────────────────
async def run_investigation(
    payload_id: str,
    org_id: str,
    user_question: str,
    cfg: SBNInvestigationConfig,
) -> AsyncGenerator[RoundResult, None]:
    """
    Async generator yielding one RoundResult per round.
    Mirrors the 5-phase DyTopo loop adapted for single-investigator + tool-selection:

      Phase 1  investigator_node  → query, summary, confidence
      Phase 2  semantic_match     → activated_tools, tool_scores  (cosine, no LLM)
      Phase 3  tool_dispatch      → tool_results  (parallel HTTP, no LLM)
      Phase 4  manager_node       → halt decision + next round_goal  (deterministic)
      Phase 5  synthesise_node    → root_cause, fix, ops_summary, eng_trace  (on halt only)

    Final RoundResult on halt carries populated root_cause / ops_summary / eng_trace.
    """
    # Ensure tool embeddings are ready (only needed for real semantic matching)
    if not cfg.mock_mode and not _TOOL_KEY_EMBEDDINGS:
        precompute_tool_embeddings()

    memory: list[MemoryEntry] = []
    round_goal = (
        f"Determine what happened during routing for document {payload_id} "
        f"belonging to org {org_id}."
    )
    prev_activated_tools: list[str] = []
    prev_tool_results: dict[str, str] = {}
    tool_scores_per_round: list[dict[str, float]] = []   # for eng_trace

    for round_t in range(cfg.t_max + 1):   # +1 so cap fires cleanly on round==t_max

        # ── Phase 1: Investigator node ────────────────────────────────────────
        if cfg.mock_mode:
            inv = await _mock_investigator(round_t)
        else:
            inv = await _investigator_llm(round_t, round_goal, memory, cfg)

        investigator_query   = inv.get("query", "")
        investigator_summary = inv.get("summary", "")
        confidence           = float(inv.get("confidence", 0.0))

        # ── Phase 2: Semantic match (no LLM) ─────────────────────────────────
        if cfg.mock_mode:
            # In mock mode we know which tools the worked example activates per round
            r = min(round_t, 2)
            mock_tools_for_round = list(_MOCK_TOOLS.get(str(r), {}).keys())
            activated_tools = mock_tools_for_round
            # Produce plausible scores for audit log (consistent with design doc §10)
            _score_map = {
                0: {"FlowLogTool": 0.82, "KafkaOffsetTool": 0.61, "RoutingDecisionTool": 0.29, "OrgConfigTool": 0.21},
                1: {"FlowLogTool": 0.31, "KafkaOffsetTool": 0.28, "RoutingDecisionTool": 0.88, "OrgConfigTool": 0.71},
                2: {"FlowLogTool": 0.19, "KafkaOffsetTool": 0.17, "RoutingDecisionTool": 0.44, "OrgConfigTool": 0.89},
            }
            tool_scores = _score_map.get(r, {t: 0.0 for t in TOOL_KEYS})
        else:
            activated_tools, tool_scores = match_tools(investigator_query, cfg.tau_edge)

        tool_scores_per_round.append(tool_scores)

        # ── Phase 3: Tool dispatch (parallel, no LLM) ────────────────────────
        if activated_tools:
            if cfg.mock_mode:
                tool_results = await _mock_tool_dispatch(round_t)
            else:
                tool_results = await _real_tool_dispatch(
                    activated_tools, payload_id, org_id, cfg
                )
            # Append to memory (summary filled next round by investigator)
            for tool_name, result in tool_results.items():
                memory.append(MemoryEntry(
                    round=round_t,
                    tool=tool_name,
                    result=result,
                    summary=investigator_summary,
                ))
        else:
            tool_results = {}

        # ── Phase 4: Manager — deterministic halt rules ───────────────────────
        halt, halt_reason, next_goal = _evaluate_manager(
            round_t=round_t,
            confidence=confidence,
            activated_tools=activated_tools,
            tool_results=tool_results,
            prev_activated_tools=prev_activated_tools,
            prev_tool_results=prev_tool_results,
            investigator_summary=investigator_summary,
            cfg=cfg,
        )

        # ── Phase 5: Synthesise on halt ───────────────────────────────────────
        root_cause = suggested_fix = ops_summary = eng_trace = None
        if halt:
            if cfg.mock_mode:
                root_cause   = _MOCK_SYNTHESIS["root_cause"]
                suggested_fix = _MOCK_SYNTHESIS["suggested_fix"]
                ops_summary  = _MOCK_SYNTHESIS["ops_summary"]
                eng_trace    = _MOCK_SYNTHESIS["eng_trace"]
            else:
                synth = await _synthesise_llm(
                    payload_id=payload_id,
                    org_id=org_id,
                    user_question=user_question,
                    memory=memory,
                    halt_reason=halt_reason,
                    confidence=confidence,
                    tool_scores_per_round=tool_scores_per_round,
                    cfg=cfg,
                )
                root_cause    = synth.get("root_cause", "")
                suggested_fix = synth.get("suggested_fix", "")
                ops_summary   = synth.get("ops_summary", "")
                eng_trace     = synth.get("eng_trace", "")

        result = RoundResult(
            round=round_t,
            round_goal=round_goal,
            investigator_query=investigator_query,
            investigator_summary=investigator_summary,
            confidence=confidence,
            activated_tools=activated_tools,
            tool_scores=tool_scores,
            tool_results=tool_results,
            halt=halt,
            halt_reason=halt_reason,
            root_cause=root_cause,
            suggested_fix=suggested_fix,
            ops_summary=ops_summary,
            eng_trace=eng_trace,
        )

        prev_activated_tools = activated_tools
        prev_tool_results = tool_results

        yield result

        if halt:
            break

        round_goal = next_goal
