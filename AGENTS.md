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

## Current M2 status and next task

- M0 implementation PR #10 and foundation PR #9 have been merged.
- M1 oracle PR #11 was merged into its parent branch.
- M1 dataset PR #12 was merged into its parent branch. Confirm that all M1
  commits are propagated to `main` before opening independent branches.
- M2 issue #4 implements **restricted JSON tensor patching**, immutable revisions,
  named checkpoints, exact rollback and probing.

Review/merge the M2 PR only after CI passes and the previous branch stacking
is resolved. **After M2 is merged**, proceed to M3 issue #5 (agent feedback
experiments) and #6 (supervised baseline). Keep weight-programmed edits distinct
from gradient descent. Never run hidden-test evaluations within the editing
agent's accessible process, and always record the actual model identity rather
than assuming Astra access.

## PR handoff format
Summary; linked issue; behavior demonstrated; tests and exact commands; artifacts/metrics; assumptions; known risks; next issue.
