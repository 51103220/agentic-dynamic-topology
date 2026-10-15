# SBN Investigation Engine — Evaluation Plan

## Goal

Measure whether applying DyTopo principles to the SBN InvestigationLoopAgent
produces better tool selection and root cause accuracy than a naive baseline
that calls all tools every round with no dynamic routing.

---

## What "with vs without DyTopo" means in the SBN context

The SBN engine (`api/sbn_investigation_engine.py`) uses three DyTopo mechanisms.
The ablation removes each one independently:

| Mechanism | With DyTopo (current engine) | Without |
|---|---|---|
| Semantic topology | `match_tools()`: cosine(Q, K) > τ activates tools | Call all 4 tools every round |
| Dynamic round goal | Manager sets next `round_goal` each round | Fixed goal string every round |
| Memory accumulation | `memory: list[MemoryEntry]` grows and is fed to investigator | Investigator sees only current round's tool output |

---

## Ablation Variants

| Variant | Description |
|---|---|
| `full` | All 3 DyTopo mechanisms active (current engine) |
| `no_semantic` | All tools called every round; no cosine matching |
| `no_goal` | Fixed goal: "find root cause of {payload_id}" every round; no manager update |
| `no_memory` | Investigator sees only the current round's results; no accumulated memory |

Controlled via a new `ablation: str` field on `SBNInvestigationConfig`.

---

## QA Dataset

4 scenarios, each with known ground truth. Written by hand — these are the
labelling effort that makes the evaluation meaningful.

| # | Scenario | Ground Truth Tools | Ground Truth Root Cause | Ground Truth Fix |
|---|---|---|---|---|
| 1 | INV-78432 PostURL blank | FlowLogTool, RoutingDecisionTool, OrgConfigTool | PostURL blank in OrgPreferences since 2026-09-14 | Configure PostURL in OrgPreferences, resubmit |
| 2 | INV-99001 Kafka never consumed | FlowLogTool, KafkaOffsetTool | Message produced but consumer group lagged; never delivered downstream | Check consumer group lag, retry offset |
| 3 | INV-55210 routing_rule_matched=false | RoutingDecisionTool | No valid routing rule matched for buyer org | Fix routing rule configuration for org |
| 4 | INV-30047 OrgConfig recently changed | OrgConfigTool | OrgPreferences last_modified too recent — misconfiguration | Revert OrgConfig change made on last_modified date |

---

## Metrics

### Tool Selection (per round, then aggregated across all rounds per scenario)

```
precision = |activated ∩ ground_truth| / |activated|
recall    = |activated ∩ ground_truth| / |ground_truth|
F1        = 2 * precision * recall / (precision + recall)
```

Baseline (`no_semantic`) always achieves recall=1.0 but precision = |GT|/4,
since all 4 tools are called regardless. DyTopo should outperform on F1.

### Root Cause Accuracy

Two scoring methods:

1. **Keyword overlap** (offline, no API):
   ```python
   score = |pred_tokens ∩ gt_tokens| / |gt_tokens|
   ```

2. **LLM judge** (preferred, needs API key):
   - Model: `claude-4.5-haiku` (cheapest credible judge)
   - Prompt: "Does this predicted root cause match the ground truth? Answer yes or no."
   - Score: 1 if yes, 0 if no
   - Fallback to keyword overlap if no API key available

### Efficiency

```
rounds_to_halt        avg rounds across scenarios per variant
tools_called_total    total tool calls across all rounds per scenario
halt_reason_dist      {confidence, exhausted, cap, repeat} counts per variant
```

---

## Notebook: `notebooks/colab_10_sbn_investigation_eval.ipynb`

### Cell structure

| Cell | Content |
|---|---|
| 1 | Imports + path setup |
| 2 | QA dataset — list of 4 scenario dicts |
| 3 | Ablation config factory — returns `SBNInvestigationConfig` for each variant |
| 4 | Runner — `run_all_variants(scenario)` runs all 4 variants, collects `RoundResult` lists |
| 5 | Tool precision/recall/F1 scorer |
| 6 | Root cause scorer (keyword overlap + optional LLM judge) |
| 7 | Results aggregation — DataFrame: variant × metric |
| 8 | Results table display + interpretation notes |

### Key design constraints

- **Mock mode only** — no LLM or HTTP calls for tool dispatch; ablation variants
  differ only in routing logic, not in tool output content. This makes results
  deterministic and reproducible.
- **LLM judge is opt-in** — controlled by `USE_LLM_JUDGE = False` flag at top
  of notebook; defaults to keyword overlap so the notebook runs fully offline.
- **Ablation wired into engine** — `SBNInvestigationConfig.ablation` field
  controls branching inside `run_investigation()`; no duplicate loop code.

---

## Implementation Steps

- [ ] 1. Add `ablation: str = "full"` field to `SBNInvestigationConfig`
- [ ] 2. Branch `no_semantic`, `no_goal`, `no_memory` inside `run_investigation()`
- [ ] 3. Add mock data for scenarios 2, 3, 4 to `sbn_investigation_engine.py`
- [ ] 4. Build `notebooks/colab_10_sbn_investigation_eval.ipynb` (cells 1–8 above)
- [ ] 5. Run notebook, verify results table renders correctly
- [ ] 6. (Optional) Wire LLM judge with `claude-4.5-haiku`

---

## Expected Results (hypothesis)

| Variant | Tool F1 | Root Cause Accuracy | Rounds | Tools Called |
|---|---|---|---|---|
| `full` | highest | highest | fewest | fewest |
| `no_semantic` | lowest precision | similar or lower | same or more | most (4× every round) |
| `no_goal` | similar | lower | more | similar |
| `no_memory` | drops after round 1 | lowest | most | similar |

The hypothesis is that memory accumulation has the largest impact on root cause
accuracy (without it the investigator re-asks the same questions), and semantic
matching has the largest impact on tool economy (without it every round
incurs 4 Java API calls regardless of relevance).
