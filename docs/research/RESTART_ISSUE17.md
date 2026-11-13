# Restart instructions for the next Codex session

You are continuing Open Weight Chess issue #17 in
`/home/thomas/project/chess/owc`. The user explicitly authorized committing,
pushing, creating a PR and updating the issue, and asked the next session to
finish publication without sandbox limits. **This is an archival/publication
task, not a new experiment.** Do not start another Astra invocation or
Stockfish labeling campaign to fill missing results.

## First actions

1. Read `AGENTS.md`, `docs/DEVELOPMENT_PLAN.md`, this file,
   `docs/research/LOCAL_ARTIFACT_INVENTORY.md`, and
   `docs/research/M3_FIELD_TRIAL_REPORT.md`.
2. Inspect `git status`, branch/remotes and all staged/unstaged changes before
   editing. Preserve every existing local file. Current branch is
   `experiment/astra-m3-pilot`, based on
   `8417fbb939dc465aca269aef80dcd7cad5bc940e`; all work is currently
   **uncommitted**. The earlier sandbox denied `.git/index.lock` writes.
3. Verify that local ignored artifacts remain present, then run:

   ```bash
   .venv/bin/ruff check .
   .venv/bin/pytest -q
   .venv/bin/python scripts/research/reproduce_metrics.py --check
   .venv/bin/python scripts/research/validate_artifacts.py
   git diff --check
   ```

   Last observed results: Ruff passed, **65 pytest tests passed**, six
   derived report files replayed, and 20 local originals verified. Three
   generated test files were declared unverified and were not inspected.

## What actually happened

- One independent non-interactive Microsoft Foundry Responses nonce request
  returned HTTP 200 from `gpt-6-astra`, response ID
  `resp_02fd1c4045f5ea08006af77b5754688195aaad69b1d470884b`, with
  18 input and 11 output tokens. **The original raw provider JSON was not
  saved.** This proves a connection observation, not a weight edit.
- `codex exec -m gpt-6-astra` failed before inference because of local
  refresh-token and Azure tool-declaration errors. No original CLI log was
  archived.
- The E0 seed-11 patch attempt invoked the new adapter but failed at DNS
  before a provider response. It disclosed zero teacher labels, returned no
  patch and left zero-model hash
  `11eb14dd2b0d22f7a1d3587f2469d0983159c739f83df90fb9448bebfb279f3c`
  unchanged. It is a failed integration attempt, **not** a successful Astra
  experiment. E5/E20/E50 were not run.
- A 30-game synthetic smoke corpus produced 308 train and 28 development
  labels with Stockfish 17.1 at 1,000 requested nodes per position. Upstream
  actual node total: 336,381. On the 28 synthetic dev positions, zero,
  handcrafted and Adam MAE were 0.722732457937384,
  0.2583377147571239 and 0.704238772392273. Adam seed 11 used 100
  distinct train labels, 500 presentations and 20 updates. These are
  engineering smoke results only, with no endgames or hidden test metric.
- The generated test split is under `/tmp/owc-astra-smoke-heldout-30` with
  the same OS owner. It was not sealed. **Do not open, label, inspect, or
  publish its positions or metadata** during this publication handoff.

## Publication boundary

The public archive is already drafted in `docs/research/` and
`reports/experiments/2026-11-13-smoke-pilot/`. It contains protocol,
chronology, technical report, article claim register, reviewer next steps,
compact manifests, exact metrics/trajectory and artifact hashes. Root
`README.md` links to it. The original corpus, labels, checkpoints and
diagnostic remain in ignored `datasets/` and `runs/` and are **not** durably
archived. The raw diagnostic includes an internal inference endpoint;
`/tmp/owc_astra_connection.py` embeds that endpoint too. Do not commit or
post either raw file. Do not commit the extracted Stockfish binary or package,
credentials, `.venv`, or unlicensed PGNs. The artifact index records checksums
and explicitly states where replay is only local.

Before publication, scan staged content for credentials, internal endpoint
URLs, raw FEN/label dumps and binary files. The public summaries should
remain self-contained and candid about the missing durable raw archive.
Do not silently change original `datasets/` or `runs/` artifacts to make a
report pass. If a discrepancy is found, document it in a new derived note.

## Finish the authorized GitHub handoff

1. Review the uncommitted implementation and reports. Fix only concrete
   errors found during review. Keep issue #17 in one reviewable PR; logical
   commits may separate implementation and research archive. Do not merge.
2. Stage **only** relevant tracked/public files: `README.md`,
   `docs/ASTRA_PILOT*.md`, `docs/research/`,
   `reports/experiments/2026-11-13-smoke-pilot/`, `scripts/`, modified
   `src/owc/` files, and modified/added `tests/`. Keep ignored originals out.
3. Commit on `experiment/astra-m3-pilot`, push that branch to `origin`, and
   open a PR targeting `main`. Use the prepared body in
   `docs/research/PR_HANDOFF.md`, updating commands and CI evidence as
   needed. Link issue #17. Leave the PR unmerged for review.
4. Wait for normal GitHub Actions. Record actual CI status and URL; never
   claim CI green solely from local tests.
5. Post the prepared `docs/research/ISSUE_17_UPDATE.md` to
   `https://github.com/quaat/owc/issues/17`, adding the actual PR URL,
   commit SHA, CI status and archive links. Do not claim a successful weight
   patch or independent benchmark.
6. In the final response, give actual clickable GitHub URLs for the PR,
   issue comment, report, protocol, results, artifact index, article evidence
   and next steps. If any publication step still fails, report that exact
   blocker and the operator action required; do not invent URLs.

The previous session's `git add` failed with
`fatal: Unable to create .../.git/index.lock: Read-only file system`, and
`gh issue view 17 --repo quaat/owc` failed to connect to `api.github.com`.
The new session is intended to remove those execution barriers. The single
highest-value scientific next gate remains a fully archived, genuine Astra
patch call in a network-permitted process isolated from test data; **do not
perform it as part of this archival restart**.
