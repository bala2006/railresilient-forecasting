# Findings and decisions

## Executive finding

The project has a credible reliability-aware forecasting prototype, but the evidence supports a cautious research conclusion rather than a deployment claim.

### What worked

- The v2 redesign removed the catastrophic empty-context behavior observed in v1 by bounding quality signals and making state updates mask-aware.
- R3S-MoE v3 produced the strongest historical clean result: MAE `45.17 s`, online WIS `28.72`, and severe-delay Brier `0.0227` in one seed.
- R4S-MoE's four-expert/top-2 path is technically valid and preserves the shared expert, reliability gate, persistence anchor, and probabilistic quantile head.
- In the new one-seed v4 comparison, end-to-end offline contextual-bandit fine-tuning produced the best WIS in clean, stale, outage, and combined scenarios.
- The forecast remains close to nominal 90% coverage across the v4 variants, around 89.8%.

### What did not work consistently

- A fresh supervised four-expert/top-1 model did not beat the historical v3 result on WIS.
- Router-only RL did not improve the supervised top-1 control; it was slightly worse on clean, packet-loss, and stale-feed cases.
- Removing the balance loss was not helpful overall.
- More top-2 capacity is not automatically better: the earlier v4 control was slightly worse and slower than v3.
- CPU timing differences between concurrent runs are too noisy to support a speed claim.

## Decision table

| Question | Current answer | Confidence | Next action |
|---|---|---|---|
| Should v4 replace v3 as the default? | Not yet. | Low to moderate | Run at least 3–5 seeds and compare paired incident/day blocks. |
| Is top-2 routing justified? | Promising for mixed corruption, not proven. | Low | Report selected mass and test 4/top-1 vs 4/top-2 across seeds. |
| Did RL help? | End-to-end offline RL helped in this one run; router-only RL did not. | Low | Repeat with fixed reward, held-out validation, and multiple seeds. |
| Is this sequential railway RL? | No. It is an offline one-step contextual bandit. | High | Add a validated simulator or logged actions before making sequential RL claims. |
| Is the public demo running the checkpoint? | No. It uses an explicitly labeled deterministic fallback at the edge. | High | Keep the local Python checkpoint path for real model inference. |
| Is this ready for train operations? | No. It is advisory research software. | High | Require safety, operational, and prospective validation before any deployment discussion. |

## Recommended next experiments

1. Run 3–5 new seeds for the v3 top-1 control, v4 supervised top-1, v4 supervised top-2, and v4 end-to-end RL.
2. Freeze the reward definition before looking at test results.
3. Add a validation-only router utilization report for both primary assignments and combined top-2 selected mass.
4. Repeat latency measurements in isolated processes with fixed thread counts.
5. Compare against a stronger non-neural probabilistic baseline and, if available, official RIDE Gold data.
6. If sequential RL remains a goal, define a validated environment with causal transitions and logged-action/off-policy evaluation.

## Claim checklist

The following statements are supported:

- “A small CPU-first model can be evaluated under synthetic unreliable-feed stress.”
- “R3S-MoE was the strongest model in the completed v3 single-seed run.”
- “The first v4 top-2 control was competitive but slightly worse/slower than v3.”
- “The end-to-end offline contextual-bandit v4 run was the strongest row among five one-seed variants.”

The following statements are **not** supported:

- “R4S-MoE is proven superior.”
- “Reinforcement learning improves railway operations.”
- “The system is safe for train control, dispatch, or signaling.”
- “The results validate Japanese railway deployment.”
- “The model predicts observed passenger outcomes.”
