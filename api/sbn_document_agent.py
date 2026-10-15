"""
SBN Conversational Document Agent — full framework skeleton
============================================================
Covers the E2E architecture from "SBN AI - Agentic Framework - Doc Routing" v1.1

Layer responsibilities (enforced as structural contracts):
  Joule            renders only      (never routes or decides)
  Classifier       understands only  (never acts or calls agents)
  Coordinator      dispatches only   (never executes domain logic)
  Agents           execute only      (never render or route)
  Tools            call only         (never decide)

Intent registry (expandable — new intent = new entry here + new agent class):
  DOC_INGEST   → DocumentIngestionAgent    (Phase 1, NEW)
  DOC_CREATE   → ConversationalCreationAgent  (Phase 2 — stub, multi-turn)
  DOC_STATUS   → InvestigationLoopAgent    (implemented in sbn_investigation_engine.py)
  DOC_REPAIR   → AutoRepairAgent           (Phase 1+, NEW)
  CLARIFY      → handled inline by Coordinator (Joule asks clarifying question)

Production stack:
  LLM gateway   SAP AI Core via LangChain LLMRouter
  Embeddings    text-embedding-3-small via SAP AI Core
  Session store Redis (SAP BTP), TTL 2h  key: agent:session:{orgId}:{id}
  Persistent    HANA Cloud — SuccessfulSubmission, TenantMemory, SessionAudit
  API           FastAPI on SAP Kyma, XSUAA JWT middleware
  Streaming     SSE via GET /agent/session/{id}/progress → Joule

Cost baseline (Table §3 — Aug 2026 SAP AI Core rates):
  Investigation ReAct (claude-4.6-sonnet): $664/month  — 72% of total cost
  Confirmation summary (claude-4.5-haiku): $123/month
  Embeddings (text-embedding-3-small):     $0.30/month
  Total optimised:                         $918/month
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator

import httpx

# ── Intent taxonomy ───────────────────────────────────────────────────────────

class IntentType(str, Enum):
    DOC_INGEST   = "DOC_INGEST"
    DOC_CREATE   = "DOC_CREATE"
    DOC_STATUS   = "DOC_STATUS"
    DOC_REPAIR   = "DOC_REPAIR"
    CLARIFY      = "CLARIFY"


@dataclass
class ClassifiedIntent:
    type: IntentType
    confidence: float                         # 0.0–1.0
    structured_params: dict[str, Any]         # extracted entities (payloadId, orgId, …)
    raw_input: str


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SBNAgentConfig:
    # LLM router endpoints (via SAP AI Core)
    ai_core_base_url: str = "https://api.ai.core.sap/v2/inference/deployments"
    intent_model: str      = "gpt-4.1-nano"        # stateless, <5ms (Phase 2+; Phase 1 = rules)
    formatting_model: str  = "claude-4.5-haiku"    # OPS plain-language, gap questions
    reasoning_model: str   = "claude-4.6-sonnet"   # investigation ReAct + synthesis
    extraction_model: str  = "gpt-4o"              # Tier 3 vision/extraction (1% volume)
    embedding_model: str   = "text-embedding-3-small"

    # Java Invoice Service
    java_invoice_service_base: str = "http://java-invoice-service/private/api/v1"

    # Session / behaviour
    temperature: float = 0.0
    max_tokens_reasoning: int = 3000
    max_tokens_formatting: int = 600
    max_revalidation_cycles: int = 3             # §§ "4. Limit Re-validation to 3 Rounds"
    session_ttl_seconds: int = 7200              # 2h for creation sessions

    # Mock mode for local dev / tests (no LLM, no HTTP)
    mock_mode: bool = True


# ── Session state ─────────────────────────────────────────────────────────────

@dataclass
class AgentSession:
    session_id: str
    org_id: str
    intent: ClassifiedIntent
    phase: str = "started"                       # tracks last completed LangGraph node
    extracted_fields: dict[str, Any] = field(default_factory=dict)
    violations: list[dict] = field(default_factory=list)
    auto_enrichments: list[dict] = field(default_factory=list)
    user_answers: list[dict] = field(default_factory=list)
    revalidation_count: int = 0
    assembled_cxml: str | None = None
    final_result: dict | None = None


# ── AgentResult — returned to Coordinator and rendered by Joule ───────────────

@dataclass
class AgentResult:
    intent_type: IntentType
    session_id: str
    status: str                                  # "in_progress" | "completed" | "error"
    progress_event: str                          # short event label for SSE
    ops_summary: str | None = None               # plain-language for OPS role
    eng_trace: str | None = None                 # full trace for ENG role
    payload: dict | None = None                  # structured result data
    error: str | None = None

    def to_sse(self) -> str:
        return f"data: {json.dumps(self.payload or {'event': self.progress_event})}\n\n"


# ── Intent Classifier ─────────────────────────────────────────────────────────
# Phase 1: rule-based (zero LLM, <5ms)
# Phase 2: LLM chain with gpt-4.1-nano for NL creation intents

_INGEST_MIME_TYPES = {"application/json", "application/pdf", "image/png", "image/jpeg", "image/tiff"}
_STATUS_KEYWORDS   = {"fail", "failed", "routing", "why", "status", "didn't", "not routed", "error", "invoice"}
_REPAIR_KEYWORDS   = {"fix", "repair", "resubmit", "correct", "wrong", "invalid", "violation"}
_CREATE_KEYWORDS   = {"create", "new invoice", "make", "build", "generate", "conversational"}


def classify_intent(
    raw_input: str,
    mime_type: str | None = None,
    cfg: SBNAgentConfig | None = None,
) -> ClassifiedIntent:
    """
    Phase 1: rule-based MIME + keyword classifier.
    Returns ClassifiedIntent with confidence >= 0.70 for known patterns.
    Coordinator treats confidence < 0.70 as CLARIFY (Joule asks 1 question).
    """
    lower = raw_input.lower()

    # MIME-type check — highest confidence (structural signal)
    if mime_type and mime_type in _INGEST_MIME_TYPES:
        return ClassifiedIntent(
            type=IntentType.DOC_INGEST,
            confidence=0.99,
            structured_params={"mime_type": mime_type},
            raw_input=raw_input,
        )

    # Extract payloadId-like token (e.g. INV-78432)
    payload_match = re.search(r'\b(INV|PO|ASN|SUR)-\d+\b', raw_input, re.IGNORECASE)
    payload_id    = payload_match.group(0).upper() if payload_match else None

    # Keyword-based routing
    words = set(lower.split())
    if any(kw in lower for kw in _STATUS_KEYWORDS) and payload_id:
        return ClassifiedIntent(
            type=IntentType.DOC_STATUS,
            confidence=0.98,
            structured_params={"payloadId": payload_id},
            raw_input=raw_input,
        )
    if any(kw in lower for kw in _REPAIR_KEYWORDS):
        return ClassifiedIntent(
            type=IntentType.DOC_REPAIR,
            confidence=0.85,
            structured_params={"payloadId": payload_id},
            raw_input=raw_input,
        )
    if any(kw in lower for kw in _CREATE_KEYWORDS):
        return ClassifiedIntent(
            type=IntentType.DOC_CREATE,
            confidence=0.80,
            structured_params={},
            raw_input=raw_input,
        )

    # Low confidence — Coordinator will ask Joule to clarify
    return ClassifiedIntent(
        type=IntentType.CLARIFY,
        confidence=0.40,
        structured_params={},
        raw_input=raw_input,
    )


# ── LLM helpers ───────────────────────────────────────────────────────────────

async def _call_llm(system: str, user: str, model: str, cfg: SBNAgentConfig) -> dict:
    """
    OpenAI-compatible POST to SAP AI Core; returns parsed JSON dict.
    Falls back gracefully to a dict with 'raw' key on parse failure.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": cfg.temperature,
        "max_tokens":  cfg.max_tokens_reasoning,
        "stream":      False,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{cfg.ai_core_base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

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
    return {"raw": raw}


# ── DocumentIngestionAgent (DOC_INGEST — Path A) ──────────────────────────────
# 9-step flow:  fingerprint → template registry → extract fields → show diff →
#               confirm/answer → assemble cXML → validate → enrich+repair →
#               final confirm → POST to egress

_EXTRACTION_SYSTEM = """\
You are a document field extraction agent for B2B invoice processing.
Given raw document content, extract all known cXML invoice fields.
Output a JSON object where keys are field names (invoiceId, supplierId, buyerId,
poNumber, date, currency, totalAmount, subtotalAmount, paymentTerms, lineItems)
and values are the extracted string values. Use null for missing fields.
Respond ONLY with valid JSON."""

_CONFIRMATION_SYSTEM = """\
You are a confirmation summary agent for a B2B document submission.
Given extracted fields, auto-enrichments, and user answers, produce a plain-language
summary for an operations user showing: what was extracted, what was auto-fixed, what
the user confirmed. Keep it under 4 sentences. Respond ONLY with valid JSON:
{"ops_summary": "...", "diff": [...]}"""


@dataclass
class IngestionResult:
    session_id: str
    step: str
    extracted_fields: dict[str, Any]
    violations: list[dict]
    auto_enrichments: list[dict]
    needs_user_questions: list[str]
    confirmation_diff: list[dict] | None
    submission_summary: str | None
    eng_trace: str | None
    halt: bool
    halt_reason: str   # "confirmed" | "escalate" | "error"


_MOCK_EXTRACTION = {
    "invoiceId":      "INV-2026-001",
    "supplierId":     "SUP-12345",
    "buyerId":        "BNO-456001",
    "poNumber":       "PO-98821",
    "date":           "2026-10-01",
    "currency":       "USD",
    "totalAmount":    None,           # low confidence — needs confirmation
    "subtotalAmount": None,           # AUTO_ENRICHABLE: qty × price
    "paymentTerms":   None,           # XX: not found
    "lineItems": [
        {"lineId": "1", "qty": 5, "unitPrice": 120.0, "description": "Widget A"},
        {"lineId": "2", "qty": 10, "unitPrice": 200.0, "description": "Widget B"},
    ],
}

_MOCK_VIOLATIONS = [
    {"field": "subtotalAmount", "issue": "missing",          "fix": "qty × price = 1600",     "classification": "AUTO_ENRICHABLE"},
    {"field": "qty_line1",      "issue": "GRN: 8 units left","fix": "reduce qty 10→8",         "classification": "AUTO_ENRICHABLE"},
    {"field": "totalAmount",    "issue": "price variance 15%","fix": "PO=X, yours=Y",          "classification": "NEEDS_USER"},
]

_MOCK_CONFIRMATION = {
    "ops_summary": "INV-2026-001 submitted. Routed to BNO-456001. 2 auto-fixed. 1 confirmed by you.",
    "diff": [
        {"field": "subtotalAmount", "before": None,   "after": 1600,   "source": "AUTO: qty×price"},
        {"field": "qty_line1",      "before": 10,     "after": 8,      "source": "AUTO: GRN remainingQty=8"},
        {"field": "totalAmount",    "before": None,   "after": "kept", "source": "USER: kept current price"},
    ],
}


async def run_doc_ingest(
    session: AgentSession,
    raw_content: str,
    cfg: SBNAgentConfig,
) -> AsyncGenerator[IngestionResult, None]:
    """
    Async generator yielding one IngestionResult per major step (for SSE streaming).
    Implements Path A: DOC_INGEST — JSON/PDF/Image → validate → confirm → POST.

    Steps (each yields one result):
      1. extract     — Tier 1 template registry (deterministic) or Tier 3 LLM vision
      2. validate    — Structural API + Domain Rule APIs in parallel fan-out
      3. enrich      — AUTO_ENRICHABLE fixes applied silently
      4. ask_user    — NEEDS_USER: Joule surfaces 1 question per turn (END node)
      5. confirm     — Final diff shown; user clicks Confirm and Submit
      6. post        — POST cXML to Doc Routing Egress API

    Re-validation capped at cfg.max_revalidation_cycles (default 3) per §§ cost doc.
    """
    # Step 1 — Extract fields
    if cfg.mock_mode:
        await asyncio.sleep(0.05)
        extracted = dict(_MOCK_EXTRACTION)
    else:
        user_prompt = f"Document content:\n{raw_content[:4000]}"
        result = await _call_llm(_EXTRACTION_SYSTEM, user_prompt, cfg.extraction_model, cfg)
        extracted = result

    session.extracted_fields = extracted
    session.phase = "extract"

    yield IngestionResult(
        session_id=session.session_id,
        step="extract",
        extracted_fields=extracted,
        violations=[],
        auto_enrichments=[],
        needs_user_questions=[],
        confirmation_diff=None,
        submission_summary=None,
        eng_trace=f"extraction_tier: {'mock' if cfg.mock_mode else 'llm'}",
        halt=False,
        halt_reason="",
    )

    # Step 2+3 — Validate then auto-enrich (max re-validation cycles)
    violations = list(_MOCK_VIOLATIONS) if cfg.mock_mode else await _run_validation(extracted, session, cfg)
    auto_enrichments: list[dict] = []
    needs_user: list[str] = []

    for _ in range(cfg.max_revalidation_cycles):
        auto_enrich = [v for v in violations if v["classification"] == "AUTO_ENRICHABLE"]
        user_needed = [v for v in violations if v["classification"] == "NEEDS_USER"]
        escalate    = [v for v in violations if v["classification"] == "ESCALATE"]

        for v in auto_enrich:
            extracted[v["field"]] = v["fix"]
            auto_enrichments.append({"field": v["field"], "before": None, "after": v["fix"], "source": "AUTO"})

        violations = user_needed + escalate
        if not violations:
            break

        # Re-validate (real mode would call Java API again)
        if not cfg.mock_mode:
            violations = await _run_validation(extracted, session, cfg)
        else:
            violations = user_needed + escalate   # mock: remain stable

    session.auto_enrichments = auto_enrichments
    session.violations = violations
    session.phase = "enrich"

    yield IngestionResult(
        session_id=session.session_id,
        step="enrich",
        extracted_fields=extracted,
        violations=violations,
        auto_enrichments=auto_enrichments,
        needs_user_questions=[v["issue"] for v in violations if v["classification"] == "NEEDS_USER"],
        confirmation_diff=None,
        submission_summary=None,
        eng_trace=f"auto_enrichments: {len(auto_enrichments)} | remaining_violations: {len(violations)}",
        halt=False,
        halt_reason="",
    )

    # Step 4 — NEEDS_USER questions (each is an END node; Joule asks 1 per turn)
    needs_user_questions = [v["issue"] for v in violations if v["classification"] == "NEEDS_USER"]
    if needs_user_questions:
        session.phase = "ask_user"
        yield IngestionResult(
            session_id=session.session_id,
            step="ask_user",
            extracted_fields=extracted,
            violations=violations,
            auto_enrichments=auto_enrichments,
            needs_user_questions=needs_user_questions,
            confirmation_diff=None,
            submission_summary=None,
            eng_trace="halt: waiting for user answer",
            halt=True,
            halt_reason="needs_user",
        )
        return   # Coordinator resumes via POST /agent/session/{id}/turn with user answer

    # Step 5 — Confirmation diff
    if cfg.mock_mode:
        confirmation = dict(_MOCK_CONFIRMATION)
    else:
        conf_user = json.dumps({
            "extracted": extracted,
            "auto_enrichments": auto_enrichments,
            "user_answers": session.user_answers,
        })
        confirmation = await _call_llm(_CONFIRMATION_SYSTEM, conf_user, cfg.formatting_model, cfg)

    diff = confirmation.get("diff", auto_enrichments)
    session.phase = "confirm"

    yield IngestionResult(
        session_id=session.session_id,
        step="confirm",
        extracted_fields=extracted,
        violations=[],
        auto_enrichments=auto_enrichments,
        needs_user_questions=[],
        confirmation_diff=diff,
        submission_summary=None,
        eng_trace="awaiting user confirmation click",
        halt=True,
        halt_reason="confirm_gate",   # Coordinator pauses; resumes on user CONFIRM click
    )


async def _run_validation(
    extracted: dict,
    session: AgentSession,
    cfg: SBNAgentConfig,
) -> list[dict]:
    """
    Parallel fan-out to Doc Routing Structural Validation API +
    Domain Business Rule APIs (Invoice, PO, GRN, SES).
    All violations merged and classified: AUTO_ENRICHABLE | NEEDS_USER | ESCALATE.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            struct_task = client.post(
                f"{cfg.java_invoice_service_base}/validate/structural",
                json=extracted,
            )
            domain_task = client.post(
                f"{cfg.java_invoice_service_base}/validate/domain",
                json=extracted,
            )
            struct_resp, domain_resp = await asyncio.gather(
                struct_task, domain_task, return_exceptions=True
            )

        violations: list[dict] = []
        if not isinstance(struct_resp, Exception):
            violations += struct_resp.json().get("violations", [])
        if not isinstance(domain_resp, Exception):
            violations += domain_resp.json().get("violations", [])
        return violations
    except Exception:
        return []


# ── AutoRepairAgent (DOC_REPAIR — Path A variant with pre-loaded violations) ──
# Receives violations already identified (from a prior ingest or routing failure).
# Re-runs the enrich → user-question → re-validate loop without re-extracting.

_REPAIR_SYSTEM = """\
You are an auto-repair agent for B2B invoice violations.
Given a list of violations and the current extracted cXML fields, determine:
1. Which violations are AUTO_ENRICHABLE (derivable from system data — qty×price, GRN qty, etc.)
2. Which require user input (NEEDS_USER)
3. Which are unresolvable without ops intervention (ESCALATE)
Output JSON: {"repaired_fields": {...}, "needs_user": [...], "escalate": [...]}
Respond ONLY with valid JSON."""

_MOCK_REPAIR_RESULT = {
    "repaired_fields": {
        "subtotalAmount": 1600,
        "qty_line1": 8,
    },
    "needs_user": ["Price variance 15%: PO price differs from invoice price. Intentional?"],
    "escalate": [],
}


@dataclass
class RepairResult:
    session_id: str
    step: str
    repaired_fields: dict[str, Any]
    needs_user: list[str]
    escalate: list[str]
    revalidation_clean: bool
    ops_summary: str | None
    eng_trace: str | None
    halt: bool
    halt_reason: str   # "clean" | "needs_user" | "escalate" | "cap"


async def run_doc_repair(
    session: AgentSession,
    violations: list[dict],
    extracted_fields: dict[str, Any],
    cfg: SBNAgentConfig,
) -> AsyncGenerator[RepairResult, None]:
    """
    Async generator for DOC_REPAIR intent.
    Applies deterministic fixes first, then escalates ambiguous violations to user.
    Re-validates after each fix batch (capped at cfg.max_revalidation_cycles).
    """
    repaired: dict[str, Any] = {}
    current_violations = list(violations)

    for cycle in range(cfg.max_revalidation_cycles + 1):

        # Attempt repair
        if cfg.mock_mode:
            await asyncio.sleep(0.05)
            repair = dict(_MOCK_REPAIR_RESULT)
        else:
            user_prompt = json.dumps({
                "violations": current_violations,
                "fields": extracted_fields,
            })
            repair = await _call_llm(_REPAIR_SYSTEM, user_prompt, cfg.reasoning_model, cfg)

        repaired.update(repair.get("repaired_fields", {}))
        needs_user = repair.get("needs_user", [])
        escalate   = repair.get("escalate", [])

        # Apply repaired fields to extracted
        for k, v in repaired.items():
            extracted_fields[k] = v

        if cycle >= cfg.max_revalidation_cycles:
            yield RepairResult(
                session_id=session.session_id,
                step="repair",
                repaired_fields=repaired,
                needs_user=needs_user,
                escalate=escalate,
                revalidation_clean=False,
                ops_summary=None,
                eng_trace=f"repair_cap_reached: cycle={cycle}",
                halt=True,
                halt_reason="cap",
            )
            return

        # Re-validate
        if not cfg.mock_mode:
            current_violations = await _run_validation(extracted_fields, session, cfg)
        else:
            current_violations = []   # mock: repair succeeds

        clean = len(current_violations) == 0 and not needs_user and not escalate

        if needs_user:
            session.phase = "repair_ask_user"
            yield RepairResult(
                session_id=session.session_id,
                step="repair_ask_user",
                repaired_fields=repaired,
                needs_user=needs_user,
                escalate=escalate,
                revalidation_clean=False,
                ops_summary=None,
                eng_trace=f"repair_cycle={cycle}: {len(needs_user)} user questions",
                halt=True,
                halt_reason="needs_user",
            )
            return

        if clean:
            ops = (
                f"All violations resolved. {len(repaired)} field(s) auto-repaired. "
                f"Document is ready to resubmit."
            )
            session.phase = "repair_complete"
            yield RepairResult(
                session_id=session.session_id,
                step="repair_complete",
                repaired_fields=repaired,
                needs_user=[],
                escalate=[],
                revalidation_clean=True,
                ops_summary=ops,
                eng_trace=f"repair_cycles: {cycle + 1} | repaired_fields: {list(repaired.keys())}",
                halt=True,
                halt_reason="clean",
            )
            return


# ── Coordinator ───────────────────────────────────────────────────────────────
# Stateful (reads/writes session store).
# Selects agent from registry by IntentType.
# Manages lifecycle: start, timeout, retry, cancel.
# Publishes progress events via SSE (pushed to Joule).

_AGENT_REGISTRY: dict[IntentType, str] = {
    IntentType.DOC_INGEST: "DocumentIngestionAgent",
    IntentType.DOC_CREATE: "ConversationalCreationAgent",
    IntentType.DOC_STATUS: "InvestigationLoopAgent",
    IntentType.DOC_REPAIR: "AutoRepairAgent",
}


@dataclass
class CoordinatorResult:
    session_id: str
    agent_name: str
    status: str               # "dispatched" | "in_progress" | "completed" | "clarify" | "error"
    progress_events: list[str]
    final_result: AgentResult | None


async def coordinate(
    session: AgentSession,
    raw_content: str,
    cfg: SBNAgentConfig,
) -> AsyncGenerator[AgentResult, None]:
    """
    Coordinator async generator.
    Routes ClassifiedIntent to the correct agent and streams AgentResult events.

    Handles:
      - CLARIFY intent → yield single clarification request to Joule
      - Low confidence (<0.70) → yield clarification request
      - DOC_STATUS  → delegates to sbn_investigation_engine.run_investigation()
      - DOC_INGEST  → delegates to run_doc_ingest()
      - DOC_REPAIR  → delegates to run_doc_repair() with pre-loaded violations
      - DOC_CREATE  → stub (Phase 2 multi-turn, not implemented)
    """
    intent = session.intent

    if intent.confidence < 0.70 or intent.type == IntentType.CLARIFY:
        yield AgentResult(
            intent_type=IntentType.CLARIFY,
            session_id=session.session_id,
            status="clarify",
            progress_event="classifier:low_confidence",
            ops_summary="I'm not sure I understood your request. Could you clarify?",
        )
        return

    agent_name = _AGENT_REGISTRY.get(intent.type, "Unknown")

    yield AgentResult(
        intent_type=intent.type,
        session_id=session.session_id,
        status="dispatched",
        progress_event=f"coordinator:dispatched:{agent_name}",
        payload={"agent": agent_name, "params": intent.structured_params},
    )

    if intent.type == IntentType.DOC_STATUS:
        # Delegate to investigation engine (already fully implemented)
        from sbn_investigation_engine import run_investigation, SBNInvestigationConfig
        inv_cfg = SBNInvestigationConfig(mock_mode=cfg.mock_mode)
        params  = intent.structured_params
        async for round_result in run_investigation(
            payload_id=params.get("payloadId", "UNKNOWN"),
            org_id=session.org_id,
            user_question=intent.raw_input,
            cfg=inv_cfg,
        ):
            yield AgentResult(
                intent_type=intent.type,
                session_id=session.session_id,
                status="completed" if round_result.halt else "in_progress",
                progress_event=f"investigation:round:{round_result.round}",
                ops_summary=round_result.ops_summary,
                eng_trace=round_result.eng_trace,
                payload=round_result.to_dict(),
            )

    elif intent.type == IntentType.DOC_INGEST:
        async for step_result in run_doc_ingest(session, raw_content, cfg):
            yield AgentResult(
                intent_type=intent.type,
                session_id=session.session_id,
                status="completed" if step_result.halt else "in_progress",
                progress_event=f"ingest:{step_result.step}",
                ops_summary=step_result.submission_summary,
                eng_trace=step_result.eng_trace,
                payload={
                    "step":                 step_result.step,
                    "extracted_fields":     step_result.extracted_fields,
                    "violations":           step_result.violations,
                    "auto_enrichments":     step_result.auto_enrichments,
                    "needs_user_questions": step_result.needs_user_questions,
                    "confirmation_diff":    step_result.confirmation_diff,
                },
            )

    elif intent.type == IntentType.DOC_REPAIR:
        violations = intent.structured_params.get("violations", [])
        async for repair_result in run_doc_repair(session, violations, session.extracted_fields, cfg):
            yield AgentResult(
                intent_type=intent.type,
                session_id=session.session_id,
                status="completed" if repair_result.halt else "in_progress",
                progress_event=f"repair:{repair_result.step}",
                ops_summary=repair_result.ops_summary,
                eng_trace=repair_result.eng_trace,
                payload={
                    "step":             repair_result.step,
                    "repaired_fields":  repair_result.repaired_fields,
                    "needs_user":       repair_result.needs_user,
                    "escalate":         repair_result.escalate,
                    "halt_reason":      repair_result.halt_reason,
                },
            )

    elif intent.type == IntentType.DOC_CREATE:
        # Phase 2 — multi-turn conversational creation (stub)
        yield AgentResult(
            intent_type=intent.type,
            session_id=session.session_id,
            status="error",
            progress_event="coordinator:not_implemented",
            error="ConversationalCreationAgent is Phase 2 — not yet implemented.",
        )


# ── FastAPI entry points (thin wrappers — implemented in main.py) ─────────────
# These are documented here as contracts; actual route handlers live in main.py.
#
# POST /agent/invoke
#   Body: {raw_input: str, mime_type?: str, org_id: str}
#   Response: {session_id: str, classified_intent: ClassifiedIntent}
#   Side-effect: creates AgentSession in Redis (TTL 2h)
#
# POST /agent/session/{session_id}/turn
#   Body: {user_input: str}
#   Response: streams AgentResult events via SSE
#   Side-effect: resumes from last LangGraph checkpoint (session.phase)
#
# GET /agent/session/{session_id}/progress
#   Response: SSE stream of AgentResult.to_sse() events → consumed by Joule


# ── Smoke test ────────────────────────────────────────────────────────────────

async def _smoke_test() -> None:
    cfg = SBNAgentConfig(mock_mode=True)

    print("=== DOC_STATUS (Investigation) ===")
    intent = classify_intent("Why did INV-78432 fail routing?")
    print(f"  classified: {intent.type} confidence={intent.confidence} params={intent.structured_params}")
    session = AgentSession(
        session_id="sess-001",
        org_id="BNO-456001",
        intent=intent,
    )
    async for event in coordinate(session, "", cfg):
        if event.status in ("in_progress", "completed") and event.payload:
            p = event.payload
            print(f"  {event.progress_event}: confidence={p.get('confidence','?')} halt={p.get('halt','?')} halt_reason={p.get('halt_reason','')!r}")
        if event.status == "completed" and event.ops_summary:
            print(f"  ops_summary: {event.ops_summary}")

    print()
    print("=== DOC_INGEST ===")
    intent2 = classify_intent("Upload invoice", mime_type="application/json")
    print(f"  classified: {intent2.type} confidence={intent2.confidence}")
    session2 = AgentSession(session_id="sess-002", org_id="BNO-456001", intent=intent2)
    async for event in coordinate(session2, '{"invoiceId": "INV-2026-001"}', cfg):
        print(f"  {event.progress_event}: status={event.status} halt_reason={event.payload.get('step','') if event.payload else ''}")
        if event.status == "completed":
            break

    print()
    print("=== DOC_REPAIR ===")
    intent3 = classify_intent("Fix the violations on INV-78432")
    print(f"  classified: {intent3.type} confidence={intent3.confidence}")
    violations = [
        {"field": "subtotalAmount", "issue": "missing", "fix": "qty × price = 1600", "classification": "AUTO_ENRICHABLE"},
        {"field": "totalAmount",    "issue": "price variance 15%", "fix": "ASK USER", "classification": "NEEDS_USER"},
    ]
    intent3.structured_params["violations"] = violations
    session3 = AgentSession(session_id="sess-003", org_id="BNO-456001", intent=intent3)
    async for event in coordinate(session3, "", cfg):
        print(f"  {event.progress_event}: status={event.status}")
        if event.status == "completed":
            break

    print()
    print("=== Low-confidence → CLARIFY ===")
    intent4 = classify_intent("help me with something")
    print(f"  classified: {intent4.type} confidence={intent4.confidence}")
    session4 = AgentSession(session_id="sess-004", org_id="BNO-456001", intent=intent4)
    async for event in coordinate(session4, "", cfg):
        print(f"  {event.progress_event}: status={event.status} ops={event.ops_summary}")


if __name__ == "__main__":
    asyncio.run(_smoke_test())
