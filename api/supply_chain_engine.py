"""
Supply Chain DyTopo Engine — extends dytopo_engine.py for SAP supply chain domain.

Based on: Lu et al. 2026 — DyTopo: Dynamic Topology Routing for Multi-Agent Reasoning
Applied to: SBN AI Agentic Framework — Doc Routing (SAP BTP / Joule / cXML)

Agents mirror the SBN architecture:
  IngestAgent        ← DocumentIngestionAgent  (extract fields, fingerprint, template lookup)
  ValidatorAgent     ← ValidationOrchestrator  (structural + domain business rule checks)
  RepairAgent        ← AutoRepairAgent          (AUTO_ENRICHABLE fixes, NEEDS_USER escalation)
  InvestigatorAgent  ← InvestigationLoopAgent  (root cause ReAct loop for routing failures)

Topology evolution mirrors the 3 SBN LangGraph shared graph stages:
  Round 0  Build cXML       — dense:    all agents share document context
  Round 1  Validate+Enrich  — moderate: repair loop agents only
  Round 2  Confirm+Post     — sparse:   final handoff, is_complete=True
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import AsyncGenerator

import httpx
import numpy as np

# ── Re-use the encoder and topology math from the base engine ─────────────────
from dytopo_engine import (
    encode,
    build_adjacency,
    Agent,
    AgentOutput,
    RoundResult,
    DyTopoConfig,
    call_agent_llm,
)

# ── Supply chain role definitions (Appendix B analogue) ───────────────────────
SUPPLY_CHAIN_ROLES = ["IngestAgent", "ValidatorAgent", "RepairAgent", "InvestigatorAgent"]

SUPPLY_CHAIN_ROLE_DESCRIPTIONS = {
    "IngestAgent": (
        "Extract all document fields (invoiceID, supplier, buyer, PO, date, currency, "
        "lineItems, totalAmount) from JSON/PDF/Image input. Assign confidence scores. "
        "Fingerprint the document and look up the Template Registry for known formats. "
        "Output extracted fields with OK/!!/XX status per field."
    ),
    "ValidatorAgent": (
        "Run structural validation and domain business rule checks (Invoice, PO, GRN, SES). "
        "Classify each violation as AUTO_ENRICHABLE (agent can fix from APIs), "
        "NEEDS_USER (one clarifying question), or ESCALATE (ops flag, do not block). "
        "Output a violation summary table with suggested fixes."
    ),
    "RepairAgent": (
        "Apply AUTO_ENRICHABLE fixes silently from session APIs and GRN data. "
        "Ask the user exactly one NEEDS_USER question per turn via Joule. "
        "Re-validate after each fix (max 3 cycles). "
        "Output enriched fields with diff: original vs repaired."
    ),
    "InvestigatorAgent": (
        "Investigate routing failures via ReAct tool loop (max 10 iterations). "
        "Tools: FlowLogTool, KafkaOffsetTool, RoutingDecisionTool, OrgConfigTool. "
        "Synthesise root cause and provide plain-language fix suggestion for Ops. "
        "Only activates when routing failure is detected — silent on happy path."
    ),
}

SUPPLY_CHAIN_WORKER_PROMPT = """\
You are {name}, the {role} in a SAP supply chain document processing pipeline.
Role Description: {description}
STRICT RESPONSE FORMAT - MANDATORY
You MUST output ONLY a valid JSON object with these exact fields:
{{
"public_content": "String. Your contribution to the current stage.",
"private_content": {{"TargetRole": "Instruction"}},
"q_vector": "String. REQUIRED. What you currently need from other agents (Query).",
"k_vector": "String. REQUIRED. What you can provide to other agents (Key)."
}}"""

SUPPLY_CHAIN_MANAGER_PROMPT = """\
You are the Supply Chain Workflow Orchestrator managing a document submission pipeline.
Decide if the document processing is COMPLETE (validated, repaired, confirmed, and posted).
STRICT RESPONSE FORMAT - MANDATORY
You MUST output ONLY a valid JSON object:
{{
"public_content": "String. Pipeline status summary and next directives.",
"private_content": {{}},
"q_vector": "String. REQUIRED. What you need next.",
"k_vector": "String. REQUIRED. What you provide.",
"is_complete": Boolean,
"next_goal": "String"
}}
CRITICAL RULES:
1. Only set is_complete=True when document is validated AND violations resolved AND user confirmed.
2. Never block clean document posting on ESCALATE-class violations."""


# ── Mock goals per stage ───────────────────────────────────────────────────────
_SUPPLY_CHAIN_GOALS = [
    "Extract and classify all document fields. Fingerprint document and identify supplier template.",
    "Run structural and domain validation. Classify violations as AUTO_ENRICHABLE / NEEDS_USER / ESCALATE. Apply silent fixes.",
    "Confirm all repairs with user. Assemble final cXML. Post to Doc Routing Egress API. Generate SUR if required.",
]

# ── Mock Q/K data — crafted so τ=0.35 produces dense→moderate→sparse ──────────
# Round 0: broad overlapping queries → dense graph (all agents share document context)
# Round 1: targeted repair queries   → moderate graph (repair loop only)
# Round 2: "nothing further" queries → sparse graph   (final handoff)

_SUPPLY_CHAIN_MOCK: dict[tuple[str, int], dict] = {

    # ── Round 0: Document ingestion & initial validation ──────────────────────
    ("IngestAgent", 0): {
        "public_content": (
            "Extracted: invoiceID=INV-2026-001, supplier=ABC Corp, buyer=BNO-456001, "
            "PO=PO-98821, date=2026-10-01, currency=USD. "
            "OK: invoiceID, supplier, buyer, PO, date. "
            "!! totalAmount=1600 (low confidence — please confirm). "
            "XX paymentTerms (not found — please provide). "
            "Template fingerprint: FP-ABC-INVOICE-v3 (HIT, Tier 1 <50ms)."
        ),
        "q_vector": "I need PO validation rules and GRN remaining quantity to validate extracted fields",
        "k_vector": "I provide extracted document fields with confidence scores and template fingerprint",
        "is_complete": False,
        "next_goal": "Validate extracted fields against business rules",
    },
    ("ValidatorAgent", 0): {
        "public_content": (
            "Structural validation: PASS. "
            "Business rule violations: "
            "(1) SubtotalAmount missing — qty(5) x price(320) = 1600. AUTO_ENRICHABLE. "
            "(2) GRN remaining qty=8 vs invoice qty=10. AUTO_ENRICHABLE (reduce qty 10→8). "
            "(3) Price variance 15% vs PO price. NEEDS_USER (intentional or match PO?)."
        ),
        "q_vector": "I need extracted document fields from IngestAgent to run structural and domain validation",
        "k_vector": "I provide structural violations and business rule violations classified as AUTO_ENRICHABLE / NEEDS_USER / ESCALATE",
        "is_complete": False,
        "next_goal": "Classify violations and begin repair",
    },
    ("RepairAgent", 0): {
        "public_content": "Standing by. Awaiting violation classification from ValidatorAgent.",
        "q_vector": "I need the violation list classified by AUTO_ENRICHABLE and NEEDS_USER from ValidatorAgent",
        "k_vector": "I provide silent auto-enrichment fixes from session APIs and GRN data, and single clarifying questions for NEEDS_USER violations",
        "is_complete": False,
        "next_goal": "Await violation list",
    },
    ("InvestigatorAgent", 0): {
        "public_content": "No routing failure detected. Document processing normally. Monitoring routing pipeline.",
        "q_vector": "I need routing decision log and org config data if a routing failure is detected",
        "k_vector": "I provide root cause analysis of routing failures via FlowLog, Kafka, RoutingDecision, OrgConfig tool loop",
        "is_complete": False,
        "next_goal": "Monitor routing pipeline",
    },

    # ── Round 1: Repair loop — auto-enrich + ask user ─────────────────────────
    ("IngestAgent", 1): {
        "public_content": (
            "Fields updated from repair session. "
            "SubtotalAmount=1600 confirmed via qty(5) x price(320). "
            "qty changed 10→8 from GRN API. "
            "Waiting for user answer on price variance before locking session."
        ),
        "q_vector": "I need repair confirmation and user answer to finalize and lock session fields",
        "k_vector": "I provide locked confirmed field values ready for cXML assembly",
        "is_complete": False,
        "next_goal": "Lock fields and assemble cXML",
    },
    ("ValidatorAgent", 1): {
        "public_content": (
            "Re-validation cycle 1: "
            "SubtotalAmount AUTO-FIXED ✓. qty 10→8 AUTO-FIXED ✓. "
            "Price variance NEEDS_USER: 'Line 2 price $320.00 vs PO $275.00 (15.5% diff). Intentional or match PO?'"
        ),
        "q_vector": "I need RepairAgent to apply AUTO_ENRICHABLE fixes and report re-validation result",
        "k_vector": "I provide re-validation result after each repair cycle (max 3 cycles)",
        "is_complete": False,
        "next_goal": "Apply fixes and escalate price variance to user",
    },
    ("RepairAgent", 1): {
        "public_content": (
            "AUTO-ENRICH applied (silent): "
            "SubtotalAmount ADDED $1600. Line qty CHANGED 10→8 (GRN remainingQty=8). "
            "Joule asking user: 'Line 2 price $320.00 vs PO $275.00 (15.5% diff). Keep $320.00 or match PO $275.00?'"
        ),
        "q_vector": "I need user answer on price variance to finalize repair and trigger final re-validation",
        "k_vector": "I provide silently enriched fields diff and single NEEDS_USER clarifying question",
        "is_complete": False,
        "next_goal": "Await user answer then re-validate",
    },
    ("InvestigatorAgent", 1): {
        "public_content": "No routing failure. Document routed successfully to BNO-456001 PostURL.",
        "q_vector": "I need nothing further",
        "k_vector": "I provide routing failure root cause if routing fails",
        "is_complete": False,
        "next_goal": "Standby",
    },

    # ── Round 2: Final confirmation and post ──────────────────────────────────
    ("IngestAgent", 2): {
        "public_content": "User answered: Keep $320.00. All fields confirmed and locked in session. cXML assembled.",
        "q_vector": "I need nothing further",
        "k_vector": "I provide final confirmed cXML payload ready for egress posting",
        "is_complete": False,
        "next_goal": "Complete",
    },
    ("ValidatorAgent", 2): {
        "public_content": "Re-validation cycle 2: CLEAN — no violations remaining. Document PASSED. Ready to post.",
        "q_vector": "I need nothing further",
        "k_vector": "I provide final validation PASS confirmation — document is clean",
        "is_complete": True,
        "next_goal": "DONE",
    },
    ("RepairAgent", 2): {
        "public_content": (
            "Submission summary: INV-2026-001 submitted. Routed to BNO-456001. "
            "2 auto-fixed (SubtotalAmount, qty). 1 confirmed by user (price $320.00). "
            "SUR generated. Validation: PASSED."
        ),
        "q_vector": "I need nothing further",
        "k_vector": "I provide final submission confirmation and SUR generation status",
        "is_complete": True,
        "next_goal": "DONE",
    },
    ("InvestigatorAgent", 2): {
        "public_content": "Document INV-2026-001 routed successfully. No investigation required.",
        "q_vector": "I need nothing further",
        "k_vector": "I provide routing success confirmation",
        "is_complete": True,
        "next_goal": "DONE",
    },
}

# ── Routing failure scenario mock (alternative flow) ─────────────────────────
_ROUTING_FAILURE_MOCK: dict[tuple[str, int], dict] = {
    ("IngestAgent", 0): {
        "public_content": "Extracted: invoiceID=INV-78432, supplier=XYZ Ltd, buyer=BNO-456001. Template: MISS — Tier 2 OCR fallback.",
        "q_vector": "I need routing status confirmation after document submission",
        "k_vector": "I provide extracted document fields and submission payload",
        "is_complete": False,
        "next_goal": "Submit and monitor routing",
    },
    ("ValidatorAgent", 0): {
        "public_content": "Validation PASSED. No violations. Document clean for posting.",
        "q_vector": "I need routing confirmation that document reached buyer egress",
        "k_vector": "I provide clean validation result — no violations",
        "is_complete": False,
        "next_goal": "Confirm routing success",
    },
    ("RepairAgent", 0): {
        "public_content": "No repairs needed. Document is clean.",
        "q_vector": "I need nothing further",
        "k_vector": "I provide repair status — no action required",
        "is_complete": False,
        "next_goal": "Standby",
    },
    ("InvestigatorAgent", 0): {
        "public_content": (
            "ROUTING FAILURE DETECTED for INV-78432. "
            "Iteration 1: FlowLogTool — FAILED at RoutingPublishStep. "
            "Iteration 2: KafkaOffsetTool — message never reached egress. "
            "Iteration 3: RoutingDecisionTool — method=null, PostURL lookup 404. "
            "Iteration 4: OrgConfigTool(BNO-456001) — PostURL blank. "
            "ROOT CAUSE: Buyer BNO-456001 has no active PostURL configured in OrgPreferences."
        ),
        "q_vector": "I need OrgConfig data for buyer BNO-456001 to confirm PostURL configuration",
        "k_vector": "I provide routing failure root cause analysis and plain-language fix for Ops",
        "is_complete": False,
        "next_goal": "Report root cause to Ops",
    },
    ("IngestAgent", 1): {
        "public_content": "Document re-queued pending buyer PostURL configuration fix.",
        "q_vector": "I need nothing further",
        "k_vector": "I provide re-queue confirmation once PostURL is fixed",
        "is_complete": False,
        "next_goal": "Await PostURL fix",
    },
    ("ValidatorAgent", 1): {
        "public_content": "Document was clean — routing issue is infrastructure, not validation.",
        "q_vector": "I need nothing further",
        "k_vector": "I provide validation clean confirmation",
        "is_complete": False,
        "next_goal": "Standby",
    },
    ("RepairAgent", 1): {
        "public_content": "No document repair needed. Routing failure is org config issue.",
        "q_vector": "I need nothing further",
        "k_vector": "I provide no-repair confirmation",
        "is_complete": False,
        "next_goal": "Standby",
    },
    ("InvestigatorAgent", 1): {
        "public_content": (
            "Fix: Ask buyer admin to configure PostURL in OrgPreferences, then resubmit. "
            "OPS message: 'Invoice INV-78432 failed routing. Reason: Buyer BNO-456001 has no active PostURL. "
            "Fix: Ask buyer admin to configure PostURL in OrgPreferences, then resubmit.' "
            "[Resubmit] [Escalate to Buyer Admin]"
        ),
        "q_vector": "I need nothing further",
        "k_vector": "I provide final root cause report and actionable fix for Ops",
        "is_complete": True,
        "next_goal": "DONE",
    },
    ("IngestAgent", 2): {
        "public_content": "Awaiting PostURL fix before resubmission.",
        "q_vector": "I need nothing further",
        "k_vector": "I provide resubmission readiness once config is fixed",
        "is_complete": True,
        "next_goal": "DONE",
    },
    ("ValidatorAgent", 2): {
        "public_content": "Standby — document valid, awaiting routing fix.",
        "q_vector": "I need nothing further",
        "k_vector": "I provide validation status",
        "is_complete": True,
        "next_goal": "DONE",
    },
    ("RepairAgent", 2): {
        "public_content": "No action required.",
        "q_vector": "I need nothing further",
        "k_vector": "I provide repair status",
        "is_complete": True,
        "next_goal": "DONE",
    },
}


# ── Mock agent call for supply chain ──────────────────────────────────────────
async def call_supply_chain_mock(
    agent: Agent,
    goal: str,
    round_t: int,
    scenario: str = "happy_path",
) -> dict:
    await asyncio.sleep(0.05)
    r = min(round_t, 2)
    table = _ROUTING_FAILURE_MOCK if scenario == "routing_failure" else _SUPPLY_CHAIN_MOCK
    key = (agent.role, r)
    return dict(table.get(key, {
        "public_content": f"[{agent.role} round {r}] Processing complete.",
        "q_vector": "I need nothing further",
        "k_vector": f"I provide {agent.role.lower()} domain expertise",
        "is_complete": r >= 2,
        "next_goal": "Continue" if r < 2 else "DONE",
    }))


# ── Supply chain DyTopo config helper ─────────────────────────────────────────
def supply_chain_config(
    tau: float = 0.35,
    max_rounds: int = 5,
    mock_mode: bool = True,
    llm_base_url: str = "http://localhost:11434/v1",
    llm_model: str = "qwen3:8b",
) -> DyTopoConfig:
    """Returns a DyTopoConfig pre-configured for supply chain task type."""
    return DyTopoConfig(
        task_type="supply_chain",
        n_agents=4,
        tau=tau,
        max_rounds=max_rounds,
        mock_mode=mock_mode,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
    )


# ── Algorithm 1 for supply chain ──────────────────────────────────────────────
async def run_supply_chain(
    document: str,
    cfg: DyTopoConfig,
    scenario: str = "happy_path",
) -> AsyncGenerator[RoundResult, None]:
    """
    Async generator yielding one RoundResult per round.
    Implements Algorithm 1 for supply chain document processing.

    Args:
        document: Raw document description or cXML/JSON content
        cfg:      DyTopoConfig with task_type='supply_chain'
        scenario: 'happy_path' | 'routing_failure'
    """
    agents = [Agent(id=i, role=role) for i, role in enumerate(SUPPLY_CHAIN_ROLES)]

    for a in agents:
        a.memory.append(f"Document: {document}")

    goals = _SUPPLY_CHAIN_GOALS
    current_goal = goals[0]
    final_answer: str | None = None

    for round_t in range(cfg.max_rounds):

        # ── Phase 1: Single-Pass Inference ────────────────────────────────────
        if cfg.mock_mode:
            outputs_raw = await asyncio.gather(*[
                call_supply_chain_mock(a, current_goal, round_t, scenario)
                for a in agents
            ])
        else:
            outputs_raw = await asyncio.gather(*[
                call_agent_llm(a, current_goal, round_t, [], cfg)
                for a in agents
            ])

        # ── Phase 2: Topology Induction ───────────────────────────────────────
        queries = [o.get("q_vector", "") for o in outputs_raw]
        keys    = [o.get("k_vector", "") for o in outputs_raw]
        A = build_adjacency(queries, keys, cfg.tau)

        # ── Phase 3 + 4: Memory Update ────────────────────────────────────────
        agent_outputs: list[AgentOutput] = []
        for i, (a, raw) in enumerate(zip(agents, outputs_raw)):
            in_neighbors = [j for j in range(len(agents)) if A[j, i] == 1]
            pub = raw.get("public_content", "")
            a.memory.append(f"Round {round_t} self: {pub[:200]}")
            for j in in_neighbors:
                priv = outputs_raw[j].get("public_content", "")
                a.memory.append(f"Round {round_t} from {agents[j].role}: {priv[:200]}")

            agent_outputs.append(AgentOutput(
                id=i,
                role=a.role,
                public_content=pub,
                q_vector=queries[i],
                k_vector=keys[i],
                in_neighbors=in_neighbors,
            ))

        # ── Phase 5: Completion check ─────────────────────────────────────────
        is_complete = False
        if cfg.mock_mode:
            r = min(round_t, 2)
            is_complete = r >= 2
            next_goal = goals[round_t + 1] if round_t + 1 < len(goals) else current_goal
            if is_complete:
                # pick the most informative agent output as final answer
                for ao in agent_outputs:
                    if len(ao.public_content) > 80:
                        final_answer = ao.public_content
                        break
                if not final_answer:
                    final_answer = agent_outputs[0].public_content
        else:
            # real LLM: check any agent's is_complete flag
            is_complete = any(o.get("is_complete", False) for o in outputs_raw)
            next_goal = next(
                (o.get("next_goal", current_goal) for o in outputs_raw if o.get("is_complete")),
                current_goal,
            )
            if is_complete:
                final_answer = next(
                    (o.get("public_content", "") for o in outputs_raw if o.get("is_complete")),
                    "",
                )

        result = RoundResult(
            round=round_t,
            goal=current_goal,
            agents=agent_outputs,
            adjacency=A.tolist(),
            is_complete=is_complete,
            final_answer=final_answer if is_complete else None,
        )

        yield result

        if is_complete:
            break

        current_goal = next_goal
