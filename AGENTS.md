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

## Current status and next Codex task

- M0 (#1): first runnable laboratory passed CI, implementation PR #10 was merged into the planning branch.
- The foundation planning PR #9 must be merged into `main` before the following stacked PRs are retargeted.
- M1 oracle (#2) lives in PR #11, based on the planning branch.
- M1 dataset (#3) lives in PR #12, based on the oracle branch.
- Do **not** present those PRs as merged to `main` until GitHub confirms this.

**Next task after the M1 review gates pass:** implement M2 issue #4
(auditable weight programming) in a fresh branch from the then-current `main`.
Build `inspect_weights`, `apply_patch`, `probe`, checkpoint, and exact
rollback, with validated sparse/low-rank/structured tensor edits. Every edit
requires before/after hashes and tests demonstrating changed behavior.
Keep agent edits separate from gradient-based baseline training; do not start
Stockfish feedback experiments until patch audit and held-out access controls
have passed review.

If M1 is not yet merged, prioritize review feedback, fixes and CI evidence
on PRs #9, #11 and #12 before beginning M2.

## PR handoff format
Summary; linked issue; behavior demonstrated; tests and exact commands; artifacts/metrics; assumptions; known risks; next issue.
