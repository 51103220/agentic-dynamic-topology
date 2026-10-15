# SBN DyTopo Engine — Architecture

How DyTopo (Lu et al. 2026, arXiv:2602.06039) is applied inside the SBN `InvestigationLoopAgent` (`DOC_STATUS` intent).

Read alongside:
- [`api/dytopo_engine.py`](dytopo_engine.py) — original multi-agent research benchmark engine
- [`api/sbn_investigation_engine.py`](sbn_investigation_engine.py) — adapted single-investigator SBN engine
- [`plan.md`](../plan.md) — ablation evaluation plan

---

## 1. Concept Map: DyTopo Paper → SBN Domain

| DyTopo paper (general) | SBN InvestigationLoopAgent |
|---|---|
| N peer worker agents `A_i`, each reasoning independently | 1 LLM investigator + K static tools; investigator reasons, tools only return data |
| Query descriptor `s_q^(i,t)` — "what agent i needs this round" | `investigator_query` — one-sentence question output each round (Phase 1 LLM output) |
| Key descriptor `s_k^(i,t)` — "what agent i can provide" | `TOOL_KEYS[name]` — static string describing what each tool returns; never changes at runtime |
| Adjacency matrix `A^(t) ∈ {0,1}^NxN`, full N×N rebuilt every round | `activated_tools: list[str]` — 1×K activation vector rebuilt every round via cosine |
| In-degree cap `K_in` (Table 7, §3.3) — max 3 incoming edges per agent | Not applied — only 1 investigator; K_in only meaningful for N>1 agents |
| Aggregation order `σ(t)` (§3.4) — topological sort so providers run before consumers | Not applied — tools are independent APIs, run in parallel |
| Private message routing (§3.2.2) — `m_priv` to out-neighbors only | Not applied — tools return raw data directly to investigator; no private/public split |
| Memory `M_i^(t)` (§3.3) — each agent keeps history of neighbor messages | `memory: list[MemoryEntry]` — grows each round; investigator receives all prior tool results |
| Manager halting `y^(t)` (§3.5) — LLM manager decides `is_complete` | `_evaluate_manager()` — fully deterministic, 4 explicit rules, no LLM call (financial audit compliance) |
| Round goal `C_task^(t)` (§3.5) — manager updates goal for next round | `round_goal: str` — manager appends last `investigator_summary` to form next goal |

---

## 2. What Is Kept, Adapted, and Deliberately Dropped

### Kept (direct application)

| Mechanism | Paper ref | SBN implementation |
|---|---|---|
| Semantic topology Q/K embeddings | §3.2 Eq.1 | `match_tools()`: `cosine(Q,K) > τ`; `encode_local()` → `all-MiniLM-L6-v2` (prod: `text-embedding-3-small`) |
| Per-round topology rebuild | Alg.1 L11 | `activated_tools` rebuilt every round, not fixed at session start |
| Memory accumulation across rounds | §3.3 | `MemoryEntry` list fed to investigator each round (Phase 1 LLM context) |
| Dynamic round goal | §3.5 | `next_goal` string passed to investigator as `round_goal` |
| T_max hard cap | §3.5 C | `cfg.t_max = 5` (same as paper) |

### Adapted (structurally equivalent, simplified for N=1 investigator)

| Paper concept | Adaptation reason | SBN equivalent |
|---|---|---|
| N×N adjacency matrix | N=1 investigator + K tools | 1×K activation vector |
| Aggregation order σ | Tools are independent; no inter-tool dependency | `asyncio.gather()` parallel dispatch |
| LLM manager | Financial audit requires auditability | 4-rule deterministic `_evaluate_manager()` |
| Private message routing | Tools return raw data; no agent-to-agent routing | All tool results go directly to investigator |

### Deliberately Not Applied

| Mechanism | Why not applied in SBN |
|---|---|
| K_in cap | Only 1 investigator — no in-degree to cap |
| σ(t) sort | Tools have no dependency on each other's output |
| Private routing | Tools are read-only APIs; no side-channel data |
| Peer Q/K update | Tools have static keys; they do not reason or update descriptors each round |

---

## 3. Per-Round Loop: Side-by-Side

| Phase | DyTopo Algorithm 1 (paper) | SBN `run_investigation()` |
|---|---|---|
| **1 — Inference** | Each agent `A_i` calls LLM with role prompt + prior memory + private messages from in-neighbors ordered by `σ(t-1)` | `call_llm(investigator_system, round_goal + memory)` → `{query, summary, confidence}` |
| **2 — Topology induction** | Compute `Q_vecs`, `K_vecs`; `R_mat = Q_vecs @ K_vecs.T`; `A_raw = (R_mat > τ)`; apply `K_in` cap | `embed(investigator_query)` → `q_vec`; `scores = {tool: dot(q_vec, key_emb)}`; `activated = [t if score > τ]` |
| **3 — Ordering** | `σ(t) = build_aggregation_order(A)` — Kahn's sort (DAG) or greedy heuristic (cyclic) | `asyncio.gather(*[call(tool) for tool in activated])` — parallel, no ordering needed |
| **4 — Routing & memory** | Route private messages along A; update each agent's memory `M_i` with in-neighbor private messages ordered by `σ(t)` | `_evaluate_manager(...)` → `(halt, halt_reason, next_goal)`; append tool results to memory |
| **5 — Manager control** | Manager LLM decides `is_complete`; sets next `C_task^(t+1)`; if complete → `final_answer` | If halt: `call_llm(synthesise_system, memory + halt_reason)` → `{root_cause, suggested_fix, ops_summary, eng_trace}` |

---

## 4. Key Parameter Mapping

| DyTopo paper | `dytopo_engine.py` | `sbn_investigation_engine.py` |
|---|---|---|
| τ (threshold) | `DyTopoConfig.tau = 0.3` | `SBNInvestigationConfig.tau_edge = 0.35` |
| T_max | `DyTopoConfig.max_rounds` | `SBNInvestigationConfig.t_max = 5` |
| K_in | `DyTopoConfig.k_in = 3` | not applicable (N=1) |
| N agents | `DyTopoConfig.n_agents = 4` | fixed: 1 investigator + 4 tools |
| Confidence halt | `is_complete` (LLM) | `confidence_threshold = 0.90` (rule) |
| Embedding model | `all-MiniLM-L6-v2` (local) | `all-MiniLM-L6-v2` (dev) / `text-embedding-3-small` (prod) |
| LLM model | `qwen3:8b` (local) | `qwen3:8b` (dev) / `claude-4.6-sonnet` (prod) |

---

## 5. Topology Geometry Comparison

**DyTopo — N=4 agents (full peer graph):**

```
A0 ──→ A1
 ↑  ↘   ↓
A3 ←── A2

Full N×N matrix rebuilt every round.
Edges can form cycles; σ(t) handles ordering.
K_in cap limits fan-in to 3.
```

**SBN — 1 investigator + 4 tools (bipartite):**

```
investigator ──→ FlowLogTool
             ──→ KafkaOffsetTool        (activated this round)
             ──→ RoutingDecisionTool
             ──→ OrgConfigTool

Bipartite: investigator→tools only.
No tool→tool edges (they are APIs).
No K_in cap needed (only 1 receiver).
No ordering needed (tools are parallel).
```

---

## 6. Why the Adaptation Is Valid

DyTopo's theoretical contribution is the per-round semantic Q/K matching that makes the communication topology **data-dependent** — the right agents talk to each other based on what information is actually needed, not a fixed graph.

In the SBN case, "agents" are domain APIs (tools). They do not reason and cannot update their own key descriptors. But the semantic matching principle applies identically: the investigator emits a query Q each round, tools have static key descriptors K, and `cosine(Q,K) > τ` determines which tools are activated. The topology is rebuilt every round based on what the investigator currently needs.

The three core DyTopo benefits are preserved:

1. **Dynamic routing** — tool selection changes round-by-round as the investigation narrows (early rounds: FlowLog/Kafka; later: RoutingDecision/OrgConfig)
2. **Semantic matching** — no hardcoded tool selection logic; the LLM's natural-language query is the routing signal
3. **Memory-gated progress** — the manager halts only when accumulated evidence reaches the confidence threshold, not after a fixed N calls

The three dropped mechanisms (K_in, σ(t), private routing) require multiple peer agents to be meaningful. Applying them to N=1 would add code complexity with zero algorithmic benefit.

---

## 7. Full RAG Architecture Across the Stack

Three embedding/RAG layers sit at different points in the investigation loop. Each solves a distinct problem. All share one embedding model (`text-embedding-3-small` via SAP AI Core) and one vector store (HANA Cloud `REAL_VECTOR` columns).

### Vector Stores (HANA Cloud)

| Store | Schema | Populated by |
|---|---|---|
| `AGENT_VIOLATION_HISTORY` | `org_id TEXT`, `payload_id TEXT`, `root_cause TEXT`, `resolution TEXT`, `count INT`, `embedding REAL_VECTOR` | Engine after every resolved investigation (grows over time) |
| `SAP_DOMAIN_KB` | `concept TEXT`, `text TEXT`, `embedding REAL_VECTOR` | You, once offline from SAP docs + error code list |
| `TOOL_KEY_EMBEDDINGS` | `tool_name str`, `embedding np.ndarray` (in-memory) | `precompute_tool_embeddings()` at pod startup |

### Investigation Loop with RAG Layers

```
PRE-LOOP ── RAG Layer 1: Tenant Memory retrieval
│
│  query     = f"{user_question} org:{org_id}"
│  query_vec = embed(query)                      ← text-embedding-3-small
│  past_cases = hana_vector_search(              ← COSINE_SIMILARITY() native
│      table   = AGENT_VIOLATION_HISTORY,
│      filter  = {org_id: org_id},
│      top_k   = 3,
│  )
│  round_goal = base_goal + past_cases
│
│  Effect: LLM starts with org-specific precedent → fewer rounds needed
│  Cost:   1 embed call + 1 HANA query, once per investigation
│
└─► ROUND t ────────────────────────────────────────────────────────────
    │
    │  Phase 1  Investigator LLM                              [LLM call]
    │           input:  round_goal + memory (+ past_cases round 0 only)
    │           output: {query Q, summary, confidence}
    │
    │  Phase 2  RAG Layer 2: Tool selection (DyTopo semantic match)
    │           q_vec     = embed(query Q)                ← embedding only
    │           score[t]  = cosine(q_vec, TOOL_KEY_EMBEDDINGS[t])
    │           activated = [t for t if score[t] > τ]
    │
    │           This IS the DyTopo topology induction step.
    │           No LLM call — pure vector arithmetic, <1ms
    │
    │  Phase 3  Tool dispatch — parallel HTTP to Java Invoice Service
    │           asyncio.gather(*[call(tool) for tool in activated])
    │           │
    │           └─► RAG Layer 3: Domain KB enrichment (on demand)
    │                 for each unknown term in tool_results:
    │                   term_vec   = embed(term)
    │                   definition = hana_vector_search(SAP_DOMAIN_KB, top_k=1)
    │                 enriched_results = tool_results + definitions
    │
    │                 Effect: LLM sees plain-language meaning of SAP
    │                 error codes / enums it has never seen
    │                 Cost: 1 embed + 1 HANA query per unknown term
    │
    │  Phase 4  Manager — deterministic halt             [no LLM, no embed]
    │           append enriched_results to memory
    │
    │  Phase 5  Synthesise LLM (on halt only)                  [LLM call]
    │           input:  full memory (enriched) + halt_reason
    │           output: root_cause, fix, ops_summary, eng_trace
    │
    └─► POST-HALT: write to AGENT_VIOLATION_HISTORY
          embed(root_cause) → store with org_id, payload_id
          feeds Layer 1 for future investigations of same org
```

### RAG Layer Summary

| Layer | Name | When | Vector store | Effect |
|---|---|---|---|---|
| 1 | Tenant Memory retrieval | Pre-loop, once per investigation | `AGENT_VIOLATION_HISTORY` (HANA) | LLM gets org-specific precedent → fewer rounds |
| 2 | Tool selection (DyTopo) | Phase 2, every round | `TOOL_KEY_EMBEDDINGS` (in-memory) | Only relevant tools called each round |
| 3 | Domain KB enrichment | Phase 3, on demand per unknown term | `SAP_DOMAIN_KB` (HANA) | LLM understands SAP error codes + enums → less hallucination |

### Embedding Model per Layer

| Layer | Dev / local | Production (SAP AI Core) |
|---|---|---|
| 1 | `all-MiniLM-L6-v2` (384d) | `text-embedding-3-small` (1536d) |
| 2 | `all-MiniLM-L6-v2` (384d) | `text-embedding-3-small` (1536d) |
| 3 | `all-MiniLM-L6-v2` (384d) | `text-embedding-3-small` (1536d) |

All three layers share one `embed()` function. Swap `SBNInvestigationConfig.embedding_model` to change all three.

### Data Flow

| | What | When |
|---|---|---|
| **Written once (by you, offline)** | `SAP_DOMAIN_KB` entries — error codes, routing enums, field definitions | Before prod deploy |
| **Written at pod startup** | `TOOL_KEY_EMBEDDINGS` — pre-computed from `TOOL_KEYS` dict | `precompute_tool_embeddings()` |
| **Written at runtime** | `AGENT_VIOLATION_HISTORY` — root_cause + resolution + org_id + embedding | After each resolved investigation |
| **Read at runtime** | Layer 1: `AGENT_VIOLATION_HISTORY` filtered by `org_id` → round 0 context | Pre-loop |
| **Read at runtime** | Layer 2: `TOOL_KEY_EMBEDDINGS` → every round | Phase 2 |
| **Read at runtime** | Layer 3: `SAP_DOMAIN_KB` → on unknown terms | Phase 3 on demand |

### Build Priority

| Priority | Layer | Action | Value |
|---|---|---|---|
| Done | 2 | Already built in `sbn_investigation_engine.py` | DyTopo tool routing |
| Next | 3 | Write 20–30 `SAP_DOMAIN_KB` entries, embed once, store in HANA | LLM stops hallucinating on SAP error codes immediately |
| Later | 1 | `AGENT_VIOLATION_HISTORY` self-populates after ~50 resolved investigations per org | Value compounds over time |

---

## 8. Ablation Evaluation

> See [`plan.md`](../plan.md) for full specification.

Three ablation variants measure the value of each DyTopo mechanism:

| Variant | What is removed | Expected degradation |
|---|---|---|
| `no_semantic` | Cosine matching removed; all 4 tools called every round | Precision drops; 4× Java API load at peak RPS |
| `no_goal` | Manager does not update `round_goal`; fixed goal string every round | Investigator loops on same question; more rounds to reach confidence |
| `no_memory` | Memory list not passed to investigator LLM; only current round visible | Investigator cannot build on prior findings; highest impact on root cause accuracy |

**Metrics:**

| Metric | Measures |
|---|---|
| `tool_precision / recall / F1` | Measured against QA ground-truth tool sets |
| `root_cause_accuracy` | LLM judge (`claude-4.5-haiku`) or token overlap |
| `rounds_to_halt` | Efficiency |
| `tools_called_total` | Java API load proxy |

Implemented in: `notebooks/colab_10_sbn_investigation_eval.ipynb` *(planned)*

RAG layers 1 and 3 introduce two additional ablation axes — `no_tenant_memory` and `no_domain_kb` — which can be added to the same notebook as additional variant columns.
