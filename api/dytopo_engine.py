"""
DyTopo engine — Algorithm 1 from Lu et al. 2026 (arXiv:2602.06039)
Pure Python, no FastAPI imports.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import AsyncGenerator

import httpx
import numpy as np

# ── Sentence encoder (loaded once, cached) ────────────────────────────────────
_encoder = None

def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def encode(texts: list[str]) -> np.ndarray:
    """Returns L2-normalised (N, 384) float32 array."""
    enc = _get_encoder()
    vecs = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype=np.float32)


# ── Data types ────────────────────────────────────────────────────────────────
@dataclass
class DyTopoConfig:
    task_type: str = "code"          # "code" | "math"
    n_agents: int = 4
    tau: float = 0.3
    max_rounds: int = 5
    mock_mode: bool = True
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:8b"
    llm_api_key: str = ""            # set for Hyperspace/OpenAI; leave empty for local vLLM
    temperature: float = 0.3
    max_tokens: int = 3000
    k_in: int = 3                    # §Table 7: max incoming neighbors per agent
    no_think: bool = False           # prepend /no_think to disable Qwen3 thinking mode


@dataclass
class Agent:
    id: int
    role: str
    memory: list[str] = field(default_factory=list)


@dataclass
class AgentOutput:
    id: int
    role: str
    public_content: str
    private_content: dict                 # routed only to out-neighbors (§3.2.1)
    q_vector: str          # natural-language query descriptor
    k_vector: str          # natural-language key descriptor
    in_neighbors: list[int] = field(default_factory=list)
    tokens_in: int = 0     # prompt tokens from API usage field
    tokens_out: int = 0    # completion tokens from API usage field


@dataclass
class RoundResult:
    round: int
    goal: str
    agents: list[AgentOutput]
    adjacency: list[list[int]]
    is_complete: bool
    final_answer: str | None = None
    tokens_in: int = 0     # total prompt tokens this round (all agents + manager)
    tokens_out: int = 0    # total completion tokens this round

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "goal": self.goal,
            "agents": [
                {
                    "id": a.id,
                    "role": a.role,
                    "public_content": a.public_content,
                    "private_content": a.private_content,
                    "q_vector": a.q_vector,
                    "k_vector": a.k_vector,
                    "in_neighbors": a.in_neighbors,
                    "tokens_in": a.tokens_in,
                    "tokens_out": a.tokens_out,
                }
                for a in self.agents
            ],
            "adjacency": self.adjacency,
            "is_complete": self.is_complete,
            "final_answer": self.final_answer,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }


# ── Topology math (Algorithm 1 lines 11-15) ───────────────────────────────────
def build_adjacency(queries: list[str], keys: list[str], tau: float) -> np.ndarray:
    """
    Returns binary N×N adjacency matrix A where A[j,i]=1 means j→i edge active.
    Edge j→i activates when key_j semantically matches query_i above threshold τ.
    """
    Q = encode(queries)   # (N, d)
    K = encode(keys)      # (N, d)
    R = Q @ K.T           # (N, N): R[i,j] = q̂_i · k̂_j
    A = (R > tau).astype(int)
    np.fill_diagonal(A, 0)    # no self-loops (1 - δ_ij)
    return A


def apply_k_in_cap(A: np.ndarray, R: np.ndarray, k_in: int) -> np.ndarray:
    """
    Enforce max in-degree K_in (Table 7): keep only the top-k_in incoming edges
    per agent ranked by relevance score R[i,j] (i=receiver, j=sender).
    """
    A = A.copy()
    N = A.shape[0]
    for i in range(N):
        incoming = np.where(A[:, i] == 1)[0]
        if len(incoming) > k_in:
            scores = R[i, incoming]          # R[i,j] = q̂_i · k̂_j  (receiver i, sender j)
            keep = incoming[np.argsort(scores)[-k_in:]]
            A[:, i] = 0
            A[keep, i] = 1
    return A


def build_aggregation_order(A: np.ndarray) -> list[int]:
    """
    §3.4: deterministic aggregation order σ(t).
    Case I (DAG): standard topological sort — provider precedes consumer.
    Case II (cyclic): greedy heuristic minimising restricted in-degree d_in(i; U).
    """
    N = A.shape[0]
    # Try Kahn's topological sort
    in_deg = A.sum(axis=0).tolist()   # in_deg[i] = number of incoming edges to i
    queue = sorted([i for i in range(N) if in_deg[i] == 0])
    order = []
    remaining = set(range(N))
    adj_list = {i: [j for j in range(N) if A[i, j] == 1] for i in range(N)}

    while queue:
        node = queue.pop(0)
        order.append(node)
        remaining.discard(node)
        for succ in sorted(adj_list[node]):
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)

    if len(order) == N:
        return order   # DAG — clean topological sort

    # Cyclic graph: greedy cycle-breaking (Eq. 10-11 in paper)
    # U = unplaced nodes; pick i* = argmin restricted in-degree d_in(i; U)
    U = sorted(remaining)
    while U:
        def restricted_in_deg(i):
            return sum(1 for j in U if A[j, i] == 1)
        i_star = min(U, key=lambda i: (restricted_in_deg(i), i))
        order.append(i_star)
        U.remove(i_star)

    return order


# ── Role definitions (Appendix B) ─────────────────────────────────────────────
CODE_ROLES = ["Developer", "Researcher", "Tester", "Designer"]
MATH_ROLES = ["ProblemParser", "Solver", "Verifier", "Manager"]

ROLE_DESCRIPTIONS = {
    "Developer":     "Write ONLY the complete Python function implementation. Output the raw code inside a ```python ... ``` block. No explanations, no descriptions — just the working code with the correct function name and signature from the problem statement.",
    "Researcher":    "Identify standard algorithms and time complexity. Output conclusions without full derivation.",
    "Tester":        "Provide critical test cases and expected results. Describe testing logic rather than full execution logs.",
    "Designer":      "Design API interfaces and class structures. Only show method signatures and type hints.",
    "ProblemParser": "Analyze the problem statement, identify known conditions, define the solving target, and break down the solving steps.",
    "Solver":        "Solve the math problem step by step. End your response with the final answer on its own line in the format: Answer: <value>. The value must be a number or expression (e.g. 42, 3/4, \\sqrt{2}).",
    "Verifier":      "Check the rationality of each derivation step, identify logical loopholes or calculation errors.",
    "Manager":       "Orchestrate the workflow. Decide if the task is COMPLETE only when solution exists AND verification passes.",
}

WORKER_SYSTEM_PROMPT = """\
You are {name}, the {role}.
Role Description: {description}
STRICT RESPONSE FORMAT - MANDATORY
You MUST output ONLY a valid JSON object with these exact fields:
{{
"public_content": "String. Your contribution to the task.",
"private_content": {{"TargetRole": "Instruction"}},
"q_vector": "String. REQUIRED. What you need next (Query).",
"k_vector": "String. REQUIRED. What you provide (Key)."
}}"""

MANAGER_SYSTEM_PROMPT = """\
You are Manager, the Workflow Orchestrator. Decide if the task is COMPLETE.
STRICT RESPONSE FORMAT - MANDATORY
You MUST output ONLY a valid JSON object with these exact fields:
{{
"public_content": "String. Status summary and next directives.",
"private_content": {{}},
"q_vector": "String. REQUIRED. What you need next (query descriptor).",
"k_vector": "String. REQUIRED. What you provide (key descriptor).",
"is_complete": Boolean,
"next_goal": "String",
"final_answer": "String. REQUIRED when is_complete=true. Copy the EXACT final answer here: for code tasks copy the complete Python function(s); for math tasks copy only the final numeric or symbolic answer (e.g. 42, 3/4, \\sqrt{{2}}). Empty string when is_complete=false."
}}
CRITICAL RULES:
1. Primary focus: Workflow Orchestrator. Decide if the task is COMPLETE.
2. Strict constraint: Only set is_complete=True if a complete solution exists AND it has been verified.
3. When is_complete=true you MUST populate final_answer with the exact solution — not a summary.
4. Analyze all agent contributions from your role's perspective."""


def _build_user_prompt(agent: Agent, goal: str, round_t: int,
                        neighbor_msgs: list[tuple[int, str, str]]) -> str:
    """Build the per-round user prompt (Algorithm 1 line 6 — S_i^(t)).
    neighbor_msgs is ordered by aggregation order σ(t) and contains private messages.
    """
    mem_block = "\n".join(agent.memory[-6:]) if agent.memory else "(empty)"
    lines = [
        f"=== Round {round_t} ===",
        f"Current goal: {goal}",
        f"Your memory:\n{mem_block}",
    ]
    if neighbor_msgs:
        lines.append("Private messages from neighbors (ordered by relevance):")
        for nid, nrole, msg in neighbor_msgs:
            lines.append(f"  [{nrole} #{nid}]: {msg[:400]}")
    return "\n".join(lines)


# ── LLM call ──────────────────────────────────────────────────────────────────
async def call_agent_llm(
    agent: Agent,
    goal: str,
    round_t: int,
    neighbor_msgs: list[tuple[int, str, str]],
    cfg: DyTopoConfig,
) -> dict:
    is_manager = agent.role == "Manager"
    system = (
        MANAGER_SYSTEM_PROMPT if is_manager
        else WORKER_SYSTEM_PROMPT.format(
            name=f"Agent{agent.id}",
            role=agent.role,
            description=ROLE_DESCRIPTIONS.get(agent.role, agent.role),
        )
    )
    if cfg.no_think:
        system = "/no_think\n" + system
    user = _build_user_prompt(agent, goal, round_t, neighbor_msgs)

    payload = {
        "model": cfg.llm_model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": False,
    }

    headers = {"Authorization": f"Bearer {cfg.llm_api_key}"} if cfg.llm_api_key else {}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{cfg.llm_base_url}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        resp_json = resp.json()
        raw = resp_json["choices"][0]["message"]["content"]
        usage = resp_json.get("usage", {})
        _tokens_in  = usage.get("prompt_tokens", 0)
        _tokens_out = usage.get("completion_tokens", 0)

    # extract JSON from potentially-markdown-wrapped response
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        blob = json_match.group()
        try:
            result = json.loads(blob)
        except json.JSONDecodeError:
            fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', blob)
            try:
                result = json.loads(fixed)
            except json.JSONDecodeError:
                result = {}
                for fname in ('answer', 'query', 'key', 'public_content', 'is_complete', 'final_answer'):
                    m = re.search(rf'"{fname}"\s*:\s*"(.*?)"(?=\s*[,}}])', blob, re.DOTALL)
                    if m:
                        result[fname] = m.group(1)
                    else:
                        m2 = re.search(rf'"{fname}"\s*:\s*(true|false|\d+)', blob)
                        if m2:
                            v = m2.group(1)
                            result[fname] = True if v == 'true' else False if v == 'false' else int(v)
                if 'public_content' not in result:
                    result['public_content'] = raw[:400]
                result.setdefault('q_vector', 'need further information')
                result.setdefault('k_vector', 'partial response')
        result['_usage'] = {'tokens_in': _tokens_in, 'tokens_out': _tokens_out}
        return result

    # No JSON object found at all
    return {
        'public_content': raw[:400],
        'q_vector': 'need further information',
        'k_vector': 'partial response',
        '_usage': {'tokens_in': _tokens_in, 'tokens_out': _tokens_out},
    }


# ── Mock mode — scripted responses (no LLM needed) ────────────────────────────
# Q/K pairs are chosen so τ=0.3 produces: Round1 dense, Round2 moderate, Round3 sparse
# matching paper Figure 4 topology evolution pattern.

_CODE_MOCK: dict[tuple[str, int], dict] = {
    # (role, round) → output dict
    ("Developer",  0): {
        "public_content": "I'll implement the solution using dynamic programming. The core data structure will be a hash map for O(1) lookups.",
        "q_vector": "I need algorithm analysis and complexity guidance from the Researcher",
        "k_vector": "I provide complete runnable Python implementation with documented functions",
        "is_complete": False, "next_goal": "Define approach and decompose problem",
    },
    ("Researcher", 0): {
        "public_content": "Standard algorithm: two-pass scan with hash map. Time O(n), Space O(n). No DP needed.",
        "q_vector": "I need the API design specification from the Designer",
        "k_vector": "I provide algorithm analysis and time complexity conclusions",
        "is_complete": False, "next_goal": "Define approach",
    },
    ("Tester",     0): {
        "public_content": "Key test cases: empty input, single element, all duplicates, negative numbers.",
        "q_vector": "I need the Developer's implementation to write specific test assertions",
        "k_vector": "I provide test case specifications and expected outputs",
        "is_complete": False, "next_goal": "Identify test coverage",
    },
    ("Designer",   0): {
        "public_content": "API: def solve(nums: list[int]) -> int. Returns -1 on empty input.",
        "q_vector": "I need algorithm requirements to finalize type hints",
        "k_vector": "I provide API interface design and class structure",
        "is_complete": False, "next_goal": "Define interfaces",
    },
    ("Developer",  1): {
        "public_content": "```python\ndef solve(nums):\n    seen = {}\n    for i, n in enumerate(nums):\n        if n in seen: return seen[n]\n        seen[n] = i\n    return -1\n```",
        "q_vector": "I need the Tester to verify my implementation handles edge cases",
        "k_vector": "I provide complete verified Python implementation ready for testing",
        "is_complete": False, "next_goal": "Verify implementation correctness",
    },
    ("Researcher", 1): {
        "public_content": "Implementation matches O(n) target. Single pass is correct.",
        "q_vector": "I need confirmation that tests pass",
        "k_vector": "I provide complexity verification and algorithm correctness analysis",
        "is_complete": False, "next_goal": "Verify efficiency",
    },
    ("Tester",     1): {
        "public_content": "Tests: solve([2,1,3,2])==0✓, solve([])==-1✓, solve([1])==-1✓. All pass.",
        "q_vector": "I need final formatted solution from Developer",
        "k_vector": "I provide test execution results and verification status",
        "is_complete": False, "next_goal": "Confirm all tests pass",
    },
    ("Designer",   1): {
        "public_content": "Type hints confirmed. Function signature is clean.",
        "q_vector": "I need test results to finalize documentation",
        "k_vector": "I provide final API documentation",
        "is_complete": False, "next_goal": "Finalize documentation",
    },
    ("Developer",  2): {
        "public_content": "```python\ndef solve(nums: list[int]) -> int:\n    \"\"\"Returns index of first duplicate or -1.\"\"\"\n    seen: dict[int, int] = {}\n    for i, n in enumerate(nums):\n        if n in seen:\n            return seen[n]\n        seen[n] = i\n    return -1\n```",
        "q_vector": "I need final sign-off from Manager",
        "k_vector": "I provide final production-ready implementation",
        "is_complete": False, "next_goal": "Deliver final solution",
    },
    ("Researcher", 2): {
        "public_content": "Solution is optimal. O(n) time, O(n) space.",
        "q_vector": "I need Manager's final assessment",
        "k_vector": "I provide final algorithm quality assessment",
        "is_complete": False, "next_goal": "Final review",
    },
    ("Tester",     2): {
        "public_content": "All test cases pass. Solution is complete.",
        "q_vector": "I need nothing further",
        "k_vector": "I provide final test verification sign-off",
        "is_complete": False, "next_goal": "Complete",
    },
    ("Designer",   2): {
        "public_content": "Documentation complete. API is clean.",
        "q_vector": "I need nothing further",
        "k_vector": "I provide final design approval",
        "is_complete": False, "next_goal": "Complete",
    },
}

_MATH_MOCK: dict[tuple[str, int], dict] = {
    ("ProblemParser", 0): {
        "public_content": "Problem: Find integer solutions to the given equation. Known conditions identified. Breaking into sub-steps: (1) simplify, (2) factor, (3) enumerate cases.",
        "q_vector": "I need the Solver to begin mathematical derivation",
        "k_vector": "I provide problem decomposition and solving plan",
        "is_complete": False, "next_goal": "Parse and decompose the problem",
    },
    ("Solver",        0): {
        "public_content": "Beginning derivation. Step 1: rearranging terms. Step 2: applying factorization identity.",
        "q_vector": "I need the problem analysis and known conditions from ProblemParser",
        "k_vector": "I provide detailed mathematical derivations and intermediate steps",
        "is_complete": False, "next_goal": "Begin derivation",
    },
    ("Verifier",      0): {
        "public_content": "Standing by to verify once derivation is complete.",
        "q_vector": "I need Solver's derivation steps to check for logical errors",
        "k_vector": "I provide logical verification and error identification",
        "is_complete": False, "next_goal": "Await derivation",
    },
    ("ProblemParser", 1): {
        "public_content": "Constraints confirmed: integer domain, positive values only. Solver's approach is aligned.",
        "q_vector": "I need Solver's intermediate results to verify constraints are respected",
        "k_vector": "I provide constraint verification and problem re-analysis if needed",
        "is_complete": False, "next_goal": "Verify constraint compliance",
    },
    ("Solver",        1): {
        "public_content": "Derivation complete. Result: x=3, y=4 is the unique solution satisfying all constraints. Answer: 5.",
        "q_vector": "I need Verifier to confirm my derivation is logically sound",
        "k_vector": "I provide complete step-by-step solution with final answer",
        "is_complete": False, "next_goal": "Verify solution",
    },
    ("Verifier",      1): {
        "public_content": "Verification: All steps are logically sound. Answer=5 checks out. No errors found.",
        "q_vector": "I need Manager to assess if solution is complete",
        "k_vector": "I provide verification confirmation and quality assurance",
        "is_complete": False, "next_goal": "Confirm verification",
    },
    ("ProblemParser", 2): {
        "public_content": "Problem fully decomposed and solution matches all constraints.",
        "q_vector": "I need nothing further",
        "k_vector": "I provide final problem-solution alignment check",
        "is_complete": False, "next_goal": "Final check",
    },
    ("Solver",        2): {
        "public_content": "Final answer confirmed: 5.",
        "q_vector": "I need nothing further",
        "k_vector": "I provide the verified final answer",
        "is_complete": True, "next_goal": "DONE",
    },
    ("Verifier",      2): {
        "public_content": "All checks passed. Solution is mathematically correct.",
        "q_vector": "I need nothing further",
        "k_vector": "I provide final verification approval",
        "is_complete": True, "next_goal": "DONE",
    },
}

_MOCK_GOALS = {
    "code": [
        "Define the initial approach and decompose the problem into roles.",
        "Confirm the Developer's implementation passes the Tester's cases.",
        "Produce the final formatted solution that matches the problem statement exactly.",
    ],
    "math": [
        "Parse the problem, identify known conditions, define the solving target.",
        "Execute derivation and verify each step for logical soundness.",
        "Produce the final verified answer.",
    ],
}

_MOCK_MANAGER: list[dict] = [
    {"is_complete": False, "next_goal": "placeholder"},
    {"is_complete": False, "next_goal": "placeholder"},
    {"is_complete": True,  "next_goal": "DONE"},
]


async def call_agent_mock(
    agent: Agent,
    goal: str,
    round_t: int,
    cfg: DyTopoConfig,
) -> dict:
    """Return scripted response — no LLM, deterministic for demo."""
    await asyncio.sleep(0.1)   # simulate tiny latency
    r = min(round_t, 2)
    table = _CODE_MOCK if cfg.task_type == "code" else _MATH_MOCK
    key = (agent.role, r)
    out = table.get(key, {
        "public_content": f"[{agent.role} round {r}] Analysis complete.",
        "q_vector": "I need further information from teammates",
        "k_vector": f"I provide {agent.role.lower()} expertise",
        "is_complete": r >= 2, "next_goal": "Continue" if r < 2 else "DONE",
    })
    return dict(out)


# ── Manager role handling ──────────────────────────────────────────────────────
def _get_roles(cfg: DyTopoConfig) -> list[str]:
    base = CODE_ROLES if cfg.task_type == "code" else MATH_ROLES
    roles = base[:cfg.n_agents]
    # ensure exactly n_agents roles (cycle if n_agents > base length)
    while len(roles) < cfg.n_agents:
        roles.append(base[len(roles) % len(base)])
    return roles


def _initial_goal(cfg: DyTopoConfig, problem: str) -> str:
    goals = _MOCK_GOALS.get(cfg.task_type, _MOCK_GOALS["code"])
    return goals[0]


# ── Private message formatter ─────────────────────────────────────────────────
def _format_private(raw: dict, role: str) -> str:
    """
    Extract the routable private message from an agent's raw output (§3.2.1).
    private_content is a dict like {"TargetRole": "Instruction"} or free-form.
    We concatenate all values; fall back to public_content if private is empty.
    """
    priv = raw.get("private_content", {})
    if isinstance(priv, dict) and priv:
        return " | ".join(str(v) for v in priv.values())
    if isinstance(priv, str) and priv.strip():
        return priv.strip()
    # fallback: public content (mock mode has no private_content)
    return raw.get("public_content", "")


# ── Algorithm 1 — full DyTopo loop ────────────────────────────────────────────
async def run_dytopo(
    problem: str,
    cfg: DyTopoConfig,
) -> AsyncGenerator[RoundResult, None]:
    """
    Async generator yielding one RoundResult per round.
    Implements Algorithm 1 from Lu et al. 2026.
    """
    roles = _get_roles(cfg)
    agents = [Agent(id=i, role=role) for i, role in enumerate(roles)]

    # initialise memory with problem statement
    for a in agents:
        a.memory.append(f"Task: {problem}")

    goals = _MOCK_GOALS.get(cfg.task_type, _MOCK_GOALS["code"])
    current_goal = goals[0]

    def _clean_answer(text: str) -> str:
        """Unwrap JSON/markdown wrappers so final_answer is always plain text or a code block."""
        if not text:
            return text
        t = text.strip()
        # unwrap ```json ... ``` blocks
        if t.startswith("```json"):
            m = re.search(r'```json\s*\n?(.*?)```', t, re.DOTALL)
            if m:
                try:
                    inner = json.loads(m.group(1))
                    t = inner.get("public_content") or inner.get("answer") or inner.get("content") or t
                except Exception:
                    pass
        # unwrap bare JSON objects {"public_content": ...}
        if t.startswith("{"):
            try:
                inner = json.loads(t)
                t = inner.get("public_content") or inner.get("answer") or inner.get("content") or t
            except Exception:
                pass
        return t.strip() if isinstance(t, str) else text

    final_answer: str | None = None
    best_primary_answer: str = ""  # tracks best Developer/Solver output across all rounds
    prev_outputs: list[dict] = []        # previous round outputs for private message passing
    prev_adjacency: list[list[int]] = [] # previous round adjacency (A[j][i]=1 → j→i)
    prev_sigma: list[int] = []           # previous round aggregation order σ(t-1)

    for round_t in range(cfg.max_rounds):
        # ── Phase 1: Single-Pass Inference (all agents in parallel) ──────────
        if cfg.mock_mode:
            outputs_raw = await asyncio.gather(*[
                call_agent_mock(a, current_goal, round_t, cfg)
                for a in agents
            ])
        else:
            # Real LLM: pass PRIVATE messages from PREVIOUS round (§3.2.2 Eq.3).
            # Agent i receives m_priv,j for each j where A[j,i]=1 in the prev round,
            # ordered by aggregation order σ(t-1) so insertion is deterministic.
            def _prev_private_msgs(agent_i: "Agent") -> list[tuple[int, str, str]]:
                if not prev_adjacency or not prev_outputs:
                    return []
                # collect senders j → i
                senders = [
                    j for j in range(len(agents))
                    if prev_adjacency[j][agent_i.id] == 1
                ]
                # order by prev aggregation order (earlier in σ = higher priority)
                senders_ordered = [j for j in prev_sigma if j in senders]
                return [
                    (j, agents[j].role,
                     _format_private(prev_outputs[j], agents[j].role))
                    for j in senders_ordered
                ]

            outputs_raw = await asyncio.gather(*[
                call_agent_llm(a, current_goal, round_t, _prev_private_msgs(a), cfg)
                for a in agents
            ])

        # ── Phase 2: Topology Induction (Algorithm 1 lines 10-16) ────────────
        queries = [o.get("q_vector", "") for o in outputs_raw]
        keys    = [o.get("k_vector", "") for o in outputs_raw]
        Q_vecs = encode(queries)
        K_vecs = encode(keys)
        R_mat  = Q_vecs @ K_vecs.T          # (N,N) relevance scores
        A_raw  = (R_mat > cfg.tau).astype(int)
        np.fill_diagonal(A_raw, 0)
        A = apply_k_in_cap(A_raw, R_mat, cfg.k_in)   # §Table 7: K_in=3 cap

        # ── Phase 3: Deterministic Aggregation Ordering (§3.4) ───────────────
        sigma = build_aggregation_order(A)             # σ(t): provider before consumer

        # ── Phase 4: Routing & Memory Update (§3.2.2 Eq.3) ──────────────────
        agent_outputs: list[AgentOutput] = []
        for i, (a, raw) in enumerate(zip(agents, outputs_raw)):
            in_neighbors = [j for j in range(len(agents)) if A[j, i] == 1]
            # order incoming by σ(t) (earlier position = higher priority)
            in_neighbors_ordered = [j for j in sigma if j in in_neighbors]

            pub  = raw.get("public_content", "")
            priv = raw.get("private_content", {})
            usage = raw.pop("_usage", {})

            # Memory update: own public + neighbors' PRIVATE messages (Eq.3)
            a.memory.append(f"Round {round_t} self: {pub[:200]}")
            for j in in_neighbors_ordered:
                priv_j = _format_private(outputs_raw[j], agents[j].role)
                a.memory.append(f"Round {round_t} from {agents[j].role}: {priv_j[:200]}")

            agent_outputs.append(AgentOutput(
                id=i,
                role=a.role,
                public_content=pub,
                private_content=priv if isinstance(priv, dict) else {},
                q_vector=queries[i],
                k_vector=keys[i],
                in_neighbors=in_neighbors_ordered,
                tokens_in=usage.get("tokens_in", 0),
                tokens_out=usage.get("tokens_out", 0),
            ))

        # ── Phase 5: Manager Control (Algorithm 1 lines 24-27) ───────────────
        # Track best primary role output across all rounds
        primary_roles = {"Developer", "Solver"}

        def _unwrap_content(text: str) -> str:
            """Strip JSON wrapper if LLM returned a dict instead of plain text."""
            t = text.strip()
            if t.startswith("{"):
                try:
                    inner = json.loads(t)
                    t = inner.get("public_content") or inner.get("answer") or inner.get("content") or text
                except Exception:
                    pass
            return t

        for ao in agent_outputs:
            if ao.role in primary_roles:
                content = _unwrap_content(ao.public_content)
                # For code tasks: prefer code blocks; always update to latest
                if "```" in content:
                    best_primary_answer = content
                elif not best_primary_answer:
                    best_primary_answer = content
                # For math tasks: always update to latest Solver output (last round = most refined)
                elif cfg.task_type == "math" and ao.role == "Solver":
                    best_primary_answer = content

        # In mock mode: use scripted completion logic
        # In real mode: manager agent decides
        is_complete = False
        if cfg.mock_mode:
            r = min(round_t, 2)
            is_complete = r >= 2
            if round_t + 1 < len(goals):
                next_goal = goals[round_t + 1]
            else:
                next_goal = current_goal
            if is_complete:
                # extract final answer from the most "complete" agent
                for ao in agent_outputs:
                    if "```" in ao.public_content or len(ao.public_content) > 80:
                        final_answer = ao.public_content
                        break
                if not final_answer:
                    final_answer = agent_outputs[0].public_content
        else:
            # find Manager agent in output
            mgr_idx = next(
                (i for i, a in enumerate(agents) if a.role == "Manager"),
                None,
            )
            if mgr_idx is not None:
                mgr_raw = outputs_raw[mgr_idx]
                is_complete = bool(mgr_raw.get("is_complete", False))
                next_goal = mgr_raw.get("next_goal", current_goal)
                if is_complete:
                    # Prefer explicit final_answer field; fall back to best primary role output
                    final_answer = _clean_answer(mgr_raw.get("final_answer", "").strip())
                    if not final_answer:
                        final_answer = _clean_answer(best_primary_answer or mgr_raw.get("public_content", ""))
            else:
                # no explicit manager — use heuristic after max_rounds
                is_complete = round_t >= cfg.max_rounds - 1
                next_goal = current_goal  # no goal progression without Manager
                if is_complete:
                    final_answer = _clean_answer(best_primary_answer or agent_outputs[0].public_content)

        # Last round safety net: always emit a final_answer so results are never empty.
        # Manager may have kept is_complete=False across all rounds — force completion.
        is_last_round = round_t >= cfg.max_rounds - 1
        if is_last_round and not final_answer:
            is_complete = True
            final_answer = _clean_answer(best_primary_answer or agent_outputs[0].public_content)

        result = RoundResult(
            round=round_t,
            goal=current_goal,
            agents=agent_outputs,
            adjacency=A.tolist(),
            is_complete=is_complete,
            final_answer=final_answer if is_complete else None,
            tokens_in=sum(ao.tokens_in for ao in agent_outputs),
            tokens_out=sum(ao.tokens_out for ao in agent_outputs),
        )

        # Store for next round's neighbor message passing (Algorithm 1)
        prev_outputs = outputs_raw
        prev_adjacency = A.tolist()
        prev_sigma = sigma

        yield result

        if is_complete:
            break

        current_goal = next_goal
