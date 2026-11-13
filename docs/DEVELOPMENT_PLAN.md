# Open Weight Chess (OWC) — Development Plan

Status: proposed plan for implementation. Owner: repository maintainers. Last updated: 2026-11-13.

## Mission

Build an open, reproducible chess-evaluation laboratory to test whether a coding/reasoning agent can **directly program neural-network weights** into a useful static chess evaluator, and how rapidly it improves using restricted Stockfish feedback. Compare that method against (1) no-feedback zero-shot construction, (2) supervised gradient learning of the **same architecture**, and (3) a non-neural handcrafted evaluation baseline. Separately test whether improvements transfer to playing chess with a common search engine.

Do **not** call the weight-editing technique "learning without training" unless the protocol identifies the exact allowed feedback/optimization. Distinguish **agent-authored structured tensor edits** from standard gradient descent and from black-box numerical search.

## Operating principles

1. Working, testable local vertical slices at each milestone; avoid speculative distributed infrastructure.
2. No hidden gradient updates in the direct-editing arm; no generic exhaustive/random weight search presented as agent reasoning.
3. Do not give a weight-programming agent access to held-out test data, test results, or hidden benchmark labels.
4. Stockfish is an **oracle**, not the engine being developed; the OWC engine must run its own independent move search and evaluator.
5. Record complete provenance: commit SHA, seeds, network spec, agent identity/model and prompt, edit history, Stockfish build/settings, hardware, elapsed wall time, token usage when available, and oracle consumption.
6. Any improvement claim requires held-out metrics with uncertainty, plus non-neural and gradient baselines.
7. No production API keys, engine binaries, or dataset dumps in git. Respect external asset licenses.

## Proposed minimal stack

- Python >= 3.12, PyTorch (CPU-first), numpy, python-chess, pytest, ruff, typed interfaces.
- Stockfish installed from an externally pinned binary/package, discovered via `STOCKFISH_PATH`.
- CLI-first local usage, JSONL/parquet datasets when appropriate, JSON run manifests.
- Reproducible dependency lock/pins for the implementation platform; CI verifies the full local smoke path without assuming Stockfish is installed unless provisioned by CI.
- UCI-compatible playing engine; optional minimalist browser board only after CLI gameplay works.

## Architecture boundaries

```text
source positions -> board/feature encoder -> static neural evaluator -> common alpha-beta search -> UCI/play
       |                                          ^
       +-> Stockfish oracle -> labeled datasets ---|
       |                     |                     |
       |                     +-> dev-set diagnostics -> weightspace agent -> validated tensor patch
       |
       +-> held-out test evaluator (isolated; read-only; never exposed to agent)
```

- `src/owc/board/`: FEN parsing via python-chess, features, perspective conventions, legal/terminal rules.
- `src/owc/oracle/`: Stockfish subprocess lifecycle, fixed budgets, deterministic labeling, dataset manifests.
- `src/owc/models/`: raw-board MLP and smaller engineered-feature MLP; independent inference.
- `src/owc/weightspace/`: typed structured patch operations, preconditions, probes, audit/rollback.
- `src/owc/experiments/`: direct programming, conventional supervised training, reproducible run driver.
- `src/owc/engine/`: negamax/alpha-beta, terminal scoring, quiescence, UCI.
- `src/owc/benchmarks/`: test evaluation, paired matches, efficiency reporting.
- `tests/`, `docs/`, `experiments/`.

## Network contracts (initial)

**Primary raw-board model:** 781 -> 32 (ReLU) -> 16 (ReLU) -> 1 (tanh); 25,569 scalar parameters. Input: 12 piece types x 64 squares (768 indicators), side to move (1), castling rights (4), legal en-passant file one-hot (8). Establish exact order/version in `models/encoding.py` and test it. Output uses **side-to-move perspective**, positive means better for side to move. A zero-initialized direct-programming model returns zero; the agent must deliberately break symmetry when constructing functionality.

**Secondary model:** limited explicit chess features (material, piece-square placement, pawn structure, king safety) defined precisely and versioned; matched architectural comparisons must not silently switch encodings.

**Reference evaluator:** deterministic material-plus-piece-square-table heuristic, separate from both neural approaches. Hardcoded rules in preprocessing count as prior knowledge and must be declared.

**Terminal positions:** handled exactly by the chess search; mate/stalemate, draws/repetition and 50/75-move logic must not be inferred from MLP score.

**Label definition:** choose and freeze ONE versioned primary teacher target per run: normalized Stockfish WDL expectation where supported, or a declared monotone bounded transform such as tanh(centipawns / 400). Always retain raw cp/mate/WDL fields, side-to-move conventions, engine settings and target transform in data records. Do not mix label types within a run.

## Milestones and vertical slices

### M0 — Reproducible laboratory and legal chess core

- Packaging, exact dependency pins/lock, `README.md` install/run examples, `AGENTS.md`, pytest and ruff, GitHub Actions.
- FEN encoding/decoding contracts; legal moves via python-chess; exact game-end handling; clear error messages.
- CLI `owc inspect --fen ...` prints board metadata/encoding summary and output from a zero-weight evaluator.
- **Gate:** fresh checkout can install, run tests, lint and inspect the standard chess starting position; deterministic expected output and zero-weights checkpoint. **This is the first Codex implementation task.**

### M1 — Stockfish oracle and datasets

- `owc oracle score --fen ... --nodes ...` and batch labeler with fixed engine binary/version, Threads=1, Hash, UCI options and fixed node budget (prefer nodes for repeatability; still record hardware/version; not promise bitwise repeatability across all systems).
- Curated position sources from legal games plus deliberately material-imbalanced/tactical/endgame positions, with provenance and license checks. Initial **illustrative target**: 10k train, 1k dev, 2k locked test.
- Split at **game/source group level before labeling**; deduplicate FENs and symmetric/transposed duplicates where feasible, prevent near-neighbor leakage; test only from sealed benchmark command.
- **Gate:** restartable generation, immutable manifests and complete round-trip scores; sanity test side-to-move symmetry and material imbalance.

### M2 — Neural evaluators and weightspace interface

- Batched inference, immutable model specification/checkpoint, CPU timing.
- `inspect_weights`, `apply_patch`, `probe`, `commit_checkpoint`, `rollback` APIs/CLI with typed restricted edits and structured sparse/low-rank/formula-based operations. No unrestricted serialized code execution inside patches.
- Automatic checks: finite weights, shapes, reproducible model hash, behavior smoke tests, audit log.
- **Gate:** deliberate agent-authored patch changes output on controlled test positions and rollback restores exact checkpoint.

### M3 — Agent experiment harness and baselines

- Model adapter identifying whether **Astra is actually available** for weight-editing; Codex controls implementation/harness and can be a separately named experimental agent when used. Never attribute a Codex run to Astra.
- Agent policies and strictly bounded dev-oracle feedback: E0 zero-shot, E1 five rounds, E2 twenty, E3 fifty. Enforce read/write separation for hidden test.
- Equal-architecture supervised gradient baseline; oracle-query-budget and elapsed-time comparisons (agent rounds are not equivalent to optimizer epochs).
- Run each main condition across >=3 independent seeds; preserve every prompt, patch, rollback and outcome.
- **Gate:** deterministic replay of agent edits, evaluated curve and separate leakage audit; no claims of improvement without held-out tests.

### M4 — Playable OWC engine

- Independent negamax/alpha-beta search with move ordering, terminal scoring and quiescence to reduce horizon failures.
- UCI implementation (`uci`, `isready`, `ucinewgame`, `position`, `go`, `stop`, `quit`), stable model checkpoint selection, perft/smoke tests.
- Paired games from prescribed opening FENs against material baseline, randomized/no-gradient and trained networks using identical search/node budgets; swap sides.
- Optional browser visualization once UCI end-to-end play is proven; user should be able to observe actual scores and compare checkpoints.
- **Gate:** completes 100 legal games without crashes/illegal moves and passes UCI acceptance tests.

### M5 — Scientific benchmarks and report

- Held-out evaluation error, pairwise preference/ranking, tactical slices, move agreement (when teacher lines exist), eval throughput, actual win/draw/loss + confidence intervals, budget and compute.
- Report **static evaluation** quality separately from game strength; measure learning trajectories against oracle queries, wall-clock and dollars (when known).
- Publish failure cases and ablations (raw vs engineered input, zero-shot vs rounds, no search vs common search), reproducibility manifest and license review.
- **Gate:** another developer can replicate selected runs and regenerate aggregate tables from manifests.

## Sequencing and review

Recommended implementation sequence: M0 -> M1 (small oracle) -> M2 (patch API) -> M3 zero-shot / 20 rounds -> M4 minimal UCI -> M5; then broaden all experiments. Each GitHub issue should fit an independently testable PR, cite exact commands executed and artifacts, and have reviewed changes before merge. Implement in small branches and report blocked dependencies rather than guessing.

## Decision register

- **D1:** Start with the raw-board 781/32/16/1 MLP, plus simple handcrafted baseline. Keep engineered-feature MLP as ablation.
- **D2:** Python-first with PyTorch/CPU. Defer GPUs, orchestration platforms, and distributed infrastructure.
- **D3:** Fixed Stockfish engine+analysis settings as teacher; versioned labels and immutable split metadata.
- **D4:** Codex is development agent; candidate direct-edit model is independently specified per experiment (Astra only when accessible/configured).
- **D5:** Initial delivery is a reproducible CLI and UCI engine, not a webapp.
- **D6:** Promotion from one milestone requires runnable evidence, not just documentation.
