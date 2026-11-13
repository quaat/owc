# OWC agent instructions

Read `docs/DEVELOPMENT_PLAN.md` before changes. This repository is an experimental lab, not a production chess strength claim.

## Goal
Develop a reproducible, CPU-first chess neural evaluator research harness that compares direct, auditable weight programming to conventional training and a handcrafted chess baseline.

## Execution protocol
1. Select the earliest unblocked GitHub issue in milestone order M0 -> M5. Scope each implementation to one reviewable PR.
2. Inspect current code and tests before editing. Never overwrite work done by another agent.
3. Implement a functional, locally testable vertical slice, not a scaffold with placeholders. Include CLI usage and tests for each new feature.
4. Use Python 3.12+, exact dependency pins/lock, pytest and ruff. Prefer python-chess for legal moves and PyTorch CPU for the initial network.
5. Execute tests/lint and report exact commands, pass/fail, and limitations in each PR description.
6. Record experiment metadata, seeds, engine versions and query budgets; ensure locked test data are inaccessible to editing agents.
7. Never use gradients or numerical black-box search in the *direct weight programming* experimental arm. Keep conventional gradient training in a separately named baseline.
8. Do not assume Astra model access. Expose adapters/configuration and label each run with the actual weight-editing agent.
9. Do not commit secrets, large datasets, binary engine executables, or unlicensed resources.
10. Do not merge automatically or close milestone work before acceptance gates pass. Explicitly flag hardware/Stockfish dependencies unavailable to CI.

## M3 development and next task

M0 through M2 were merged into `main`; M3 has two independent workstreams:
issue #5 for bounded agent-feedback protocol and #6 for supervised and
handcrafted evaluation baselines. Keep their PRs reviewable and distinct.

**Do not claim any real Astra/Codex experimental success** until an actual
provider/model adapter is configured, run and authenticated. The built-in
fixture is only for CI and has no chess intelligence. Precomputed teacher
labels are not the same as live Stockfish queries; account for the
upstream teacher-generation cost. External command adapters are not
automatically sandboxed. Use true OS/process isolation from held-out
test data; development-vs-test filenames are not an access-control boundary.

After the agent runner and baseline land on main, run controlled experiments
with representative licensed PGNs, an independent blind test service, three
seeds, and published model and compute provenance. The next playable
engine milestone is M4 issue #7.

## PR handoff format
Summary; linked issue; behavior demonstrated; tests and exact commands; artifacts/metrics; assumptions; known risks; next issue.
