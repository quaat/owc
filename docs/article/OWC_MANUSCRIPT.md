# Can a Frontier Model Program a Chess Evaluator by Editing Its Weights?
## An auditable experimental protocol and the first Open Weight Chess integration attempt

**Document status:** Preliminary technical article, evidence reviewed against OWC experiment archive dated **13 November 2026**. **Not a report of successful LLM-generated chess weights, not independently benchmarked, and not yet suitable for claims about chess strength.** Later measurements require a new version and an updated claim register.

**Project:** [Open Weight Chess](https://github.com/quaat/owc) (OWC)  
**Primary research issue:** [#17](https://github.com/quaat/owc/issues/17)  
**Source evidence:** [M3 field trial report](../research/M3_FIELD_TRIAL_REPORT.md), [protocol](../research/EXPERIMENT_PROTOCOL.md), [claim register](../research/ARTICLE_EVIDENCE.md), [structured results](../../reports/experiments/2026-11-13-smoke-pilot/results.json), and [artifact inventory](../../reports/experiments/2026-11-13-smoke-pilot/artifact-index.json).

## Abstract

Can a frontier language model construct a useful neural chess evaluator by directly setting its parameters, rather than optimizing those parameters by gradient descent? Open Weight Chess (OWC) is an open, auditable framework designed to test this question. It exposes a fixed 25,569-parameter position-evaluation network to structured, hash-validated tensor patches, and compares explicitly programmed weights with a same-architecture Adam-trained network and a handcrafted chess evaluator. Stockfish provides a versioned source of position labels. Experimental conditions distinguish a blind initial patch (E0) from subsequent rounds of bounded feedback (E5, E20, E50).

The first integration attempt on 13 November 2026 established a non-interactive connection to a Microsoft Foundry deployment labeled \`gpt-6-astra\`, but its E0 weight-editing request failed at DNS resolution before any provider response. Consequently, **no model-authored chess weight patch was obtained**, and the central hypothesis remains untested. On a 28-position synthetic development set, mean absolute error was 0.7227 for the zero network, 0.7042 for a small Adam-trained baseline, and 0.2583 for a handcrafted heuristic. These values only validate the evaluation pipeline; they are not evidence of comparative chess strength. This report describes the method, the observed failure, and the controls required before any scientific conclusion about direct weight programming is warranted.

## 1. Why experiment with weight-space programming?

Neural networks are ordinarily produced by defining an architecture and minimizing a loss over training examples. A different possibility is to treat the parameters themselves as a program: a capable model examines a network's structure and directly edits tensors to implement desired behavior. Takehi and colleagues reported exploratory examples of this approach under the name **weight-space programming** [1]. Their November 2026 report is a research blog post, not, by itself, proof that the approach will generalize to chess, to larger networks, or to cost-effective learning.

Chess is an informative test bed for three reasons. First, it has unambiguous legal states and strong computational evaluators. Second, changes in a neural evaluator can be inspected against chess-specific expectations: piece material, tempo, placement, mobility and tactical vulnerability. Third, chess exposes a distinction that is easy to overlook: **predicting the value of a position is not equivalent to choosing strong moves in a game**. A static network can mimic some engine scores and still perform poorly under search; conversely, a search algorithm may compensate for imperfect static evaluation.

This is not the first attempt to learn from a chess engine. Work on chess value and policy prediction has already explored extensive supervised distillation of Stockfish evaluations [2]. Stockfish itself uses neural evaluation based on efficiently updatable networks [3]. OWC does not claim novelty for neural chess evaluation or engine distillation. Its narrower question is whether a general-purpose model can construct and progressively improve a **small evaluator by explicit, inspectable parameter edits**, and what resource costs and failure modes that entails.

## 2. The central experiment

OWC tests two related hypotheses:

- **H1 — Construction:** Without receiving teacher scores, an LLM can program a network that responds sensibly to distinct chess positions rather than returning a constant or random score.
- **H2 — Adaptation:** With a bounded number of Stockfish feedback disclosures, the LLM can improve the network's prediction quality on positions it was not shown during editing.

H1 requires more than a valid JSON patch. The patch must change a checkpoint and establish a connected computation from chess input features to output, while passing basic behavioral tests such as material-direction and side-to-move consistency.

H2 requires more than a reduction in an agent-visible development metric. The agent's adaptation must ultimately be assessed on **frozen, independently held-out** positions, with a report of variability across seeds and a fair account of model calls, label exposures and elapsed computation. It is possible that direct edits improve quickly at first but plateau; it is also possible that the same architecture trained by Adam is substantially better. Both outcomes matter.

### 2.1 A deliberately small, fixed network

The primary evaluator is a floating-point multilayer perceptron:

$
781 \rightarrow 32 \;(\mathrm{ReLU}) \rightarrow 16 \;(\mathrm{ReLU}) \rightarrow 1 \;(\tanh).
$

It contains **25,569 trainable parameters**, including biases. Its feature representation, versioned as \`owc.raw781.v1\`, consists of twelve 64-square piece planes (768 features), a side-to-move indicator, four castling-right indicators and eight legal en-passant file indicators. Inputs use fixed board coordinates. The output lies in $[-1,1]$, with positive values denoting an advantage for the side to move.

For the direct-edit condition, every parameter starts at exactly zero. This is a strong and revealing restriction: with zero weights through all layers, the network has no active path from a piece-square input to the output. An agent that modifies only the last-layer bias can change the score without evaluating chess. A meaningful program must deliberately create nonzero paths across the three linear layers. The baseline trained by gradient descent instead uses seeded nonzero initialization, since keeping all hidden weights at zero would not be a fair way to train a ReLU network by Adam.

**Interpretation caveat:** if an agent writes conventional material or piece-square heuristics and compiles them into the network's weights, that is still an explicit program, but it represents a different contribution from discovering a novel evaluation function. OWC should preserve this distinction in patch reasoning and subsequent analyses.

### 2.2 An audit trail for every edit

The direct-edit interface accepts versioned JSON patch operations: sparse scalar \`set\`, scalar \`add\`, and a bounded \`rank_one_add\` for structured matrix modifications. Each patch declares its expected starting model hash and the affected tensor indices. The engine checks types, shapes, finite values, edit budgets and the model-hash precondition, then stages candidate edits before publishing a new revision. Rejected edits do not change the active checkpoint.

The audit records changes and failures, permitting replay of the exact edits applied. This is **application-level reproducibility**, not an assertion that files are tamper-proof against a user with filesystem access. The agent name in a JSON request is likewise not cryptographic proof of which model produced it. Genuine provider provenance must be recorded from actual model responses and an authenticated invocation route.

### 2.3 Stockfish as the reference oracle

Stockfish returns evaluations and search metadata under declared node budgets. The primary continuous target in the first OWC smoke trial was

$
y(s) = \tanh\left(\frac{\mathrm{cp}_{\mathrm{stm}}(s)}{400}\right),
$

where $\mathrm{cp}_{\mathrm{stm}}$ is the engine's centipawn assessment from the side-to-move perspective; mate scores map to signed terminal bounds. This transformation makes target values compatible with the network's tanh output. It does not turn centipawn scores into calibrated winning probabilities.

The dataset and score manifests include the engine build, binary checksum, analysis options, node budget, source hashes and source-game grouping. Determinism here means that configuration and artifacts are documented. It does **not** guarantee identical search results on all software builds and hardware.

### 2.4 Conditions and comparison arms

| Arm | Parameter construction | Teacher exposure | Intended role |
| --- | --- | --- | --- |
| Zero network | All-zero fixed checkpoint | None | Reference for error and constant-output behavior |
| Handcrafted | Fixed material and placement rule | None during this comparison | Explicit chess-prior baseline |
| Adam | Same 781→32→16→1 architecture; randomized initialization; gradients | Training labels | Conventional supervised baseline |
| E0 | One LLM-authored blind patch | No teacher targets | Direct-construction test |
| E5 / E20 / E50 | Blind patch, then 5 / 20 / 50 LLM feedback rounds | Bounded development-label disclosures | Feedback-efficiency tests |

Comparisons must distinguish **distinct teacher labels**, repeated disclosures, optimization steps, model input/output tokens, inference latency, Stockfish nodes, and wall-clock time. These are not interchangeable budget units. The cost of creating the teacher dataset exists even when a particular editing run reports zero *new* engine calls.

## 3. What actually happened on 13 November 2026

The first session's purpose was to exercise the tooling and begin an E0 trial. The data were generated from **30 synthetic random legal games**, not from a representative corpus of human or engine games. Source-game partitioning yielded 308 training positions from 22 games and 28 development positions from two games; six other games belonged to an unscored, **unsealed** test split. The development set contained four opening and 24 middlegame positions, no endgames, and many material imbalances. The dataset was suitable for integration testing, not generalization claims.

Stockfish **17.1** was configured for **1,000 requested nodes per position**, a single thread and 16 MiB hash. Its total *actual* searched nodes were **308,348** for train and **28,033** for development positions. A direct UCI labeling transport was used after the ordinary \`python-chess\` threaded interface stalled in that execution environment. This is a documented transport deviation, not a new method of chess evaluation.

### 3.1 Model connection succeeded; model editing did not

A separate non-interactive request reached the configured Foundry deployment identified as \`gpt-6-astra\` and returned HTTP 200, with a reported 18 input and 11 output tokens. The original raw provider response was not durably preserved, limiting independent verification of that connection observation.

The experiment runner then attempted **E0, seed 11**. It selected four blind development probes, disclosed **zero** teacher targets and attempted one Astra adapter invocation. The adapter failed at **DNS resolution** before receiving a provider response. Consequently:

- No model-generated tensor patch was returned or accepted.
- The model's original hash and zero-weight revision remained unchanged.
- No post-patch chess behavior or E0 prediction error exists.
- No E5, E20 or E50 experiment took place.

The observed failure is a network/runtime integration outcome. It does not establish that Astra can or cannot program a neural evaluator. The useful result is narrower: the execution environment must permit the *actual experiment subprocess* to reach Foundry, rather than relying on a successful connection test from a different environment.

### 3.2 Measured smoke-baseline results

The following are **development-only integration measurements**, not benchmark estimates.

| Evaluator | Development MAE (28 synthetic positions) | Interpretation |
| --- | ---: | --- |
| Zero-initialized RawMLP | 0.722732457937384 | Constant reference output |
| Handcrafted material/placement rule | 0.2583377147571239 | Explicit material and placement priors |
| Adam-trained RawMLP, seed 11 | 0.704238772392273 | One completed supervised smoke run |
| Astra E0 | Not measured | DNS failure before model response |
| Astra E5 | Not run | Depends on successful E0 and review gate |

The Adam run used **100 distinct teacher-labeled training positions**, five epochs, 500 label presentations and **20 optimizer updates**. Its development MAE decreased from 0.7227 at the first recorded post-epoch evaluation to 0.7042 after the fifth epoch. The original randomly initialized model's *pre-training* MAE was not recorded.

The handcrafted baseline's markedly lower development error is unsurprising for synthetic positions dominated by material differences, but the data cannot establish that the heuristic is generally better than learned evaluation. With only 28 development samples, no endgames and one Adam seed, treating these figures as estimates of chess strength would be misleading.

## 4. What this trial teaches us

The first lesson is methodological. A system that can modify model parameters, validate JSON, call a chess engine and produce a numerical score is **not yet evidence of successful weight-space programming**. Evidence begins only when a genuine model-generated patch passes validation and produces meaningful, reproducible behavior. The original patch and response must remain available for scrutiny.

The second lesson is about experimental isolation. The editing process should see only the permitted development probes and teacher feedback. The held-out evaluator must run under separate credentials or OS protections, with frozen checkpoints submitted to it. In this first trial the generated test files shared the operator's permissions, so no legitimate blind-test metric was obtained.

The third lesson is that engineering failures are not scientific counterexamples. A DNS error, an authentication issue or an event-loop stall cannot answer H1 or H2. It must be documented so that later results are not conflated with earlier failures, while keeping confidential endpoints and secrets out of public artifacts.

The fourth lesson concerns comparators. Stockfish NNUE is already a highly specialized and optimized neural evaluator [3]; OWC's small floating-point RawMLP is intentionally much simpler. Comparing the direct-edited MLP against Adam using the same architecture isolates an important part of the question, but does not by itself compare with modern chess-engine quality. Playing strength will require OWC's independent search/UCI milestone and paired games with identical search conditions.

## 5. The next decisive experiment

The immediate gate is **one fully archived, model-generated E0 patch** in a network-permitted process that cannot read sealed-test data. Before the call, freeze and hash the exact code revision, prompts, request schema and deployment configuration. Set an explicit invocation/token/latency budget and capture the raw provider response in restricted durable storage, plus a sanitized public attestation.

After that call, validate the patch against the current M2 model hash, publish the edit as a new immutable revision, and probe contrasting positions for nonconstant behavior, sensible material direction and side-to-move consistency. Record the actual parameter changes, patch acceptance/rejection and any failure. If no patch is returned, report another engineering failure—not a null chess-performance result.

Only after E0 passes should OWC run a cost-bounded E5 study, then consider E20 and E50. A scientifically informative comparison needs a licensed and representative corpus, predefined position categories, three or more independent seeds, disclosed resource budgets, and a **frozen-checkpoint evaluation performed outside the agent's access boundary**. Effect sizes and uncertainty should be reported before conclusions about advantage, parity or inefficiency.

## 6. Limitations and article status

As of this preliminary report, **H1 and H2 are untested**. There is no Astra-authored chess evaluator, no direct-edit learning curve, no independent benchmark, no robust seed comparison, and no chess Elo result. The original synthetic source corpus and model/checkpoint artifacts are held locally but not durably archived for external replication. The unsealed test split was never used for evaluation. The reported error numbers should remain labeled as smoke metrics.

This manuscript is a traceable **protocol and negative/incomplete integration report**. It can be developed into a substantive comparative article only when genuine model edits and independent outcomes are available. A future version should add the full experimental trajectories, case studies of successful and failed patches, cost-normalized comparison charts, confidence intervals, and an explicit accounting of the relationship between model reasoning and chess heuristics compiled into weights.

## Post-trial engineering update (13 November 2026; not a new experimental result)

Following independent review of the first field-trial archive, OWC's adapter and diagnostic safeguards were improved and merged into `main` via [PR #18](https://github.com/quaat/owc/pull/18), which also closed [issue #19](https://github.com/quaat/owc/issues/19). This includes canonical serialized request-size checks, compact feedback history, private raw transcript retention, pre-call request/instruction/source hashes, explicit response-model mapping and regression tests. GitHub Actions verified 74 tests and lint for the final experimental branch.

A separate isolated-process preflight reached the authenticated Foundry model listing and showed that a known synthetic test path was inaccessible from that namespace. This is **not** proof of a general sealed-test boundary: the mount configuration still exposes other portions of the host filesystem as read-only. No new model-generated chess patch, E5 feedback, hidden-test result or game-strength result was obtained.

For one future E0 attempt, the adapter now enforces a **32,000-byte complete provider-request limit** and the isolated launcher limits generation to **3,000 output tokens**. Microsoft's published [GPT-6 Astra Standard token prices](https://azure.microsoft.com/en-us/blog/gpt-6-astra-sol-and-luna-for-production-agents-in-microsoft-foundry/) allow a conditional estimate of **$1.038** at EU Data Zone long-context rates when using a conservative 32,000-input/3,000-output-token scenario. This estimate is neither a verified price for the actual deployment nor a billing guarantee. The actual SKU and an appropriate spending decision must be verified before the single paid call. See [the cost-gate record](../research/ISSUE_20_COST_BUDGET.md) and [issue #20](https://github.com/quaat/owc/issues/20).

This update changes the project's engineering readiness, **not** the empirical conclusions in Sections 3–6. The central H1 and H2 hypotheses remain untested.

## References and source material

1. Takehi, R., Teoh, J., Zhang, A., and Khattab, O. (2026, November 2). *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*. [Research blog post](https://rikiyat.github.io/blog/weight-space-programming/). Exploratory claims are attributed to the authors; not treated here as independently established OWC results.
2. Ruoss, A., Delétang, G., Medapati, S., et al. (2024). *Amortized Planning with Large-Scale Transformers: A Case Study on Chess*. NeurIPS 37. [Conference abstract and paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/78f0db30c39c850de728c769f42fc903-Abstract-Conference.html).
3. Official Stockfish Project. *NNUE: Principles and implementation*. [Stockfish NNUE documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html).
4. Open Weight Chess. [M3 field trial report](../research/M3_FIELD_TRIAL_REPORT.md), [protocol and deviations](../research/EXPERIMENT_PROTOCOL.md), [evidence/claim register](../research/ARTICLE_EVIDENCE.md), [machine-readable results](../../reports/experiments/2026-11-13-smoke-pilot/results.json), [artifact index](../../reports/experiments/2026-11-13-smoke-pilot/artifact-index.json), code and issue [#17](https://github.com/quaat/owc/issues/17), snapshot dated 13 November 2026.

---

*Editorial note:* This is a living Markdown manuscript maintained separately from the model experiment branch. When new findings are accepted, revise the manuscript in a fresh version with a dated change log and direct links to the supporting, immutable run artifacts. Do not silently replace the preliminary negative findings with newer outcomes.
