# Article claim-to-evidence register

This is a research planning aid, not an article success claim. All numerical
observations refer to the synthetic 28-position **development** split in
the [field trial report](M3_FIELD_TRIAL_REPORT.md). The raw artifacts are
local only; see the [artifact index](../../reports/experiments/2026-11-13-smoke-pilot/artifact-index.json).

| Proposed article claim | Evidence | Confidence | Limitation | Publishable? |
| --- | --- | --- | --- | --- |
| A non-interactive request reached the configured `gpt-6-astra` Foundry deployment | Response ID and usage in [results.json](../../reports/experiments/2026-11-13-smoke-pilot/results.json); contemporaneous tool output | Medium | Raw provider JSON not durably archived | Conditional, as engineering observation |
| A real Astra model produced a neural weight patch | No provider response or patch in [E0 manifest](../../reports/experiments/2026-11-13-smoke-pilot/e0-attempt1.manifest.json) | High confidence this trial did **not** demonstrate it | A later run may succeed | No |
| The failed E0 invocation left the model unchanged | Original event and revision hashes verified by `validate_artifacts.py` | High locally | Raw run artifacts are not in durable storage | Yes, as engineering validation with archival caveat |
| E0 disclosed no Stockfish targets | Original request has empty `teacher_feedback`; run manifest records zero disclosures | High locally | Four blind FEN probes were present | Yes, as protocol observation |
| A same-architecture Adam baseline completed | [Adam manifest](../../reports/experiments/2026-11-13-smoke-pilot/adam-seed11.manifest.json), checkpoint hash, and replay script | High locally | One seed and 100 sampled training examples | Yes, with synthetic-smoke scope |
| The handcrafted evaluator had MAE 0.2583377147571239 on this dev set | [results.json](../../reports/experiments/2026-11-13-smoke-pilot/results.json), original labels, replay code | High locally | 28 synthetic positions, no endgames | Conditional; never as general chess-strength claim |
| Adam improved over the zero network on this dev set | 0.704238772392273 versus 0.722732457937384 in [metrics.csv](../../reports/experiments/2026-11-13-smoke-pilot/metrics.csv) | Medium for this split | Single seed, synthetic selection, unmatched budgets | Conditional, narrowly phrased |
| Direct weight programming is competitive with Adam | No accepted patch or direct-model MAE | None | Hypothesis untested | No |
| Direct weight programming improves with feedback rounds | No E5/E20/E50 run | None | No trajectory | No |
| Findings generalize to unseen games | No OS-isolated held-out evaluation | None | Test directory same owner; synthetic source | No |
| OWC gained playing strength or Elo | No M4 search/UCI engine or matches | None | Static evaluation only | No |

## Evidence levels

**Confirmed engineering:** existing M2 validation and runner tests pass;
the failed E0 request was audited without changing its model; Stockfish
smoke labels and Adam checkpoint load and replay locally.

**Observed but archival-limited:** the Foundry nonce response was seen in
the prior session, with response ID and usage, but not stored as raw JSON.

**Tentative interpretation:** the handcrafted rule's low error on this
synthetic sample suggests material dominates many sampled positions. This
is a hypothesis for stratified data collection, not a causal finding.

**Unsupported:** any claim that Astra programmed useful chess weights,
outperformed a comparator, or transferred to hidden data.

## Provisional article outline

1. The methodological question: explicit tensor programming versus Adam
   under matched architecture and separately counted resources.
2. OWC's auditable patch protocol, feedback disclosure policy, and genuine
   model-identity and test-isolation requirements.
3. Integration failure modes and their effect on scientific claims; this
   field trial as a transparent negative/incomplete result.
4. A later licensed, representative multi-seed development study with
   full patch trajectories and cost accounting.
5. Frozen-checkpoint evaluation by a separate blind service, uncertainty
   estimates, and chess-position case studies.
6. Limitations: heuristic knowledge encoded into weights, data-selection
   bias, compute matching, stochastic agent outputs and playing-strength
   separation.

Potential contribution today is a reproducible **protocol and failure audit**.
The central empirical contribution requires genuine patches, independent
evaluation and replication before article submission.
