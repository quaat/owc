# Proposed OWC comparative study design

**Status:** Prospective draft for review, **not preregistered** and not evidence that experiments were executed. Freeze a dated commit and approve cost/data-isolation controls *before* treating this as a prespecified experimental plan. Related issues: [#17](https://github.com/quaat/owc/issues/17), [#19](https://github.com/quaat/owc/issues/19), [#20](https://github.com/quaat/owc/issues/20), [#21](https://github.com/quaat/owc/issues/21).

## Scientific target

**Question:** Given the same small chess-evaluation network and a documented amount of teacher feedback, can a general-purpose reasoning model explicitly construct and improve evaluator weights in a way that generalizes to unseen games, and at what model-call, token, oracle and compute cost relative to Adam training and a disclosed handcrafted rule?

**Primary endpoint for a full study:** Mean absolute error (MAE) against a versioned Stockfish-derived bounded target, evaluated on frozen, *source-game-disjoint and OS-isolated* test positions after all candidate selection has ended.

**Secondary endpoints:** Across-position variance and material-direction sanity tests, pairwise position-ranking accuracy, Spearman correlation where defensible, errors stratified by game phase, material imbalance and side to move, patch validity and rate, time/cost versus error, and consistency across seeds. No Elo or playing-strength conclusions without the separate M4 search/UCI experiment.

## Gates, not one uncontrolled run

| Gate | Method | What constitutes completion | What it does *not* establish |
| --- | --- | --- | --- |
| G0: protocol and network | Fixture, local Stockfish, restricted adapter smoke | CI green, request-size caps, no test access, traceable hashes | Genuine agent learning |
| G1: real E0 | One real Astra patch call from all-zero network | Response provenance, valid M2 patch, new model hash and nonconstant position behavior | Generalization, feedback benefit |
| G2: small E5 | Bounded feedback, with the same prompt/patch policy | Valid E0+5 accepted or explicitly recorded failed rounds, complete budget audit | Robust advantage over Adam |
| G3: representative development campaign | E0/E5; optionally E20/E50 after budget review, >=3 seeds | Complete trajectories, baselines, failure cases and frozen model-selection rule | Independent test performance |
| G4: blind evaluation | Separate OS principal/service, no agent test access | Checkpoint freeze before independent test; game-group bootstrap confidence intervals | Playing Elo |
| G5: article claims | Independent source/artifact/claim audit | Every table and graph regenerates from stored measurements and hashes | Proof of wider applicability beyond OWC |

**No automatic promotion:** G1 must succeed before G2; G2's budget must be reviewed before E20/E50. A DNS failure, invalid structured response or constant-output patch should remain a failed or noninformative outcome, not be replaced by a fixture without disclosure.

## Inputs, comparators and prior knowledge

Freeze \`owc.raw781.v1\`: 768 absolute piece-square indicators plus side-to-move, castling rights and legal en-passant file, network \`781 → 32 ReLU → 16 ReLU → 1 tanh\` (25,569 float32 parameters).

Compare:

- **Zero:** static all-zero network.
- **Handcrafted:** versioned material/placement rule with explicitly disclosed material and positional priors.
- **Adam:** same network/feature encoding, seeded nonzero initialization, fixed/declared train labels, optimizers and epochs.
- **Direct E0:** Astra constructs one patch with no teacher targets.
- **Direct E5/E20/E50:** Astra constructs initial patch then revises it using bounded disclosed development labels; no gradients, finite differences, generic random/numerical weight search or hidden helper training.

Record the **actual** provider, model/deployment identity, prompts, actor response, tool/API options, number of model calls (including rejected/failed), tokens, elapsed time, accepted patches and touched parameters. Agent knowledge about chess, feature indexing and neural-network construction is an explicitly provided prior: a high-level model with chess knowledge is not a blank-slate learner.

If Astra encodes a classical material heuristic in a neural circuit, classify the result as **heuristic compilation via weight programming**, not an automatically discovered novel strategy.

## Dataset design and licenses

Candidate source: [Lichess Open Database](https://database.lichess.org/) **standard rated-game PGNs**, which the official website identifies as **CC0**. Broadcast datasets are separately listed under CC BY-SA; do not silently mix them. Record export month/date, exact download URL or public source ID, license, SHA-256 and import/filter script SHA.

- Stream a bounded sample of PGNs instead of importing an entire huge monthly archive. Filter to standard chess, legal move history, and documented game quality/rating/time controls. If stratified by player rating, report the selection thresholds.
- Partition by **source game** and avoid crossing the same game between train, dev, or test. Deduplicate canonical board states globally. Make effort to estimate near-neighbor and opening-family overlap; do not claim these are eliminated by four-field FEN dedup alone.
- Specify phase categories *before* inspecting test outcomes. Example: a documented rule based on non-pawn material and move phase. Use the **same rule across all arms**; avoid labels that change post hoc to favor an outcome.
- Balance representation of opening/middlegame/endgame, both sides to move, near-equal and materially imbalanced positions, and tactical positions where the source permits. Preserve natural-distribution estimates separately if oversampling challenging strata.
- Produce immutable license/source/split manifests with seed, source-game IDs, hash, grouping, position counts, phase counts and exclusion reasons.
- Use one pinned local Stockfish build and fixed analysis settings to generate **primary** train/dev/test labels. Pre-existing Lichess evaluations may be useful supplementary data but differ in engine/search configuration and should not be silently combined with the primary labels.

**Suggested sizes for budgeting, not performed or approved:** a bounded *development smoke* with hundreds of positions, followed by a representative research corpus on the order of 10,000 train / 1,000 development / 2,000 independent test positions, subject to data and compute review. Effective uncertainty depends on the number of independent source games and clusters, not merely positions.

## Teacher budget and fairness

Teacher target: \(\tanh(\mathrm{cp}_{stm}/400)\), mate cases mapped to explicit bounds, preserving raw cp/mate, engine options and PV when available. Report positions evaluated, actual node count, requests and wall time for **upstream label creation**, even when the agent later consumes only precomputed feedback.

There are at least two meaningful fairness comparisons, which should be presented separately:

1. **Exposure-matched:** equal number of *distinct labeled positions* made available to the two methods, plus repeated disclosure/presentation counts. These methods still differ in prior knowledge and how labels are processed.
2. **Resource-matched:** equal or comparable wall time, model tokens / dollars and hardware resources, with differences explicitly stated.

Do **not** present one feedback round as one gradient step, or count all Stockfish-labeled train positions as “free” for the supervised baseline. E0 has zero teacher-target exposure but still consumes model inference resources and chess knowledge from pretraining.

Fix seeds, model settings and a model-selection protocol in advance. For repeated stochastic LLM calls, record provider sampling parameters where exposed and distinguish **replay** of one saved patch from **replication** with another model response.

## Test isolation and checkpoint freezing

The Codex development/publishing session may have repository write access. The *weight-editing agent* must not have access to hidden-test data, credentials or endpoints:

- Run the evaluator under a separate OS user/service/container or machine with authenticated, restricted requests. Verify actual permission denial from the editing process.
- Restrict the raw test dataset and teacher labels to the evaluator, and keep them out of the model prompt, local adapter environment, development logs and public PR.
- Freeze model hashes and run selection **before** requesting final test results.
- Run one independent blind test per frozen final candidate/seed, rather than iterating on test outcomes. Any subsequent redesign belongs to a new experimental campaign and must disclose prior test use.
- Report any accidental test exposure as a protocol violation, not as a silent correction.

The M3 runner's per-round \`dev_mae_undisclosed\` may be used for diagnostics, but track whether researchers used it for candidate selection. It is not an independent held-out score.

## Analyses and uncertainty

Primary table: mean MAE by arm and seed on the **same frozen test set**, with all failed/no-response cases visible (N/A, not zero). Include zero and handcrafted baselines plus the gradient-trained comparison. Report the number of independent seeds, position clusters, target type, evaluation budget and cost.

For uncertainty, use a *cluster bootstrap by source game* for paired model differences, recording the number of resamples and random seed; do not bootstrap individual FENs as if all same-game positions were independent. With a small number of source games, state that intervals are unstable and gather more games before significance claims.

Proposed secondary analyses: MAE by phase and material imbalance; pairwise score ordering where meaningfully defined; model-score variance; material-direction and side-to-move sanity tests; cumulative agent calls and token cost versus dev error; patch success/failure; number of edited scalars; and detailed case studies with source positions and original agent reasoning/patches where publication rights permit.

Do not compute a p-value merely to make a claim look stronger. Publish raw estimates, paired differences, uncertainty and the limitations of each analysis.

## Reporting and manuscript rules

- Give each run a unique ID and preserve original JSON requests/responses, M2 checkpoints, immutable source/label hashes, environment manifests, and token/cost evidence in a durable approved store. Save restricted transcripts under protected access and put only checksums/sanitized summaries in GitHub.
- Maintain \`docs/research/RESEARCH_LOG.md\` with failures and protocol deviations. Update \`docs/research/ARTICLE_EVIDENCE.md\` before new claims enter [the manuscript](OWC_MANUSCRIPT.md).
- Every figure should regenerate from machine-readable original or faithfully derived metrics with scripts. Separate development and test axes/labels. Do not plot failed Astra conditions as zeros.
- Classify results as: **engineering validation**, **development evidence**, **independent evidence**, **playing-strength evidence**, or **unsupported**. The 2026-11-13 smoke attempt supports only the first category.
- No article statement about “Astra learned chess” or comparative advantage without genuine accepted patches plus independent benchmark evidence.

## Pending decisions to freeze before G3

1. Exact licensed source files, sampling strata, split policy, source-game dedup method and target phase-distribution.
2. Exact Stockfish binary, teacher node budgets, target normalization, and endgame/tactical selection rules.
3. E0/E5/E20/E50 model invocation and cost ceilings; acceptable deployment/response identity mapping.
4. Supervised baseline data and compute budget schedule, number of seeds and checkpoint-selection policy.
5. Independent evaluator account, durable restricted artifact store, transcript retention and reviewer access.
6. Manuscript target audience (research note, reproducibility report or peer-reviewed article); a publication claim register should follow the strictest intended standard.

*Keep this document labeled as a proposal until a frozen version is committed before a genuine scientific campaign.*
