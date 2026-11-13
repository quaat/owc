# Chronological research log

Times are UTC on 2026-11-13. Entries marked **reconstructed** combine prior
session tool output, file timestamps and contemporaneous notes. Precise
timestamps absent from original artifacts are not invented. Original files
remain in ignored `datasets/` and `runs/`; their hashes are in the
[artifact index](../../reports/experiments/2026-11-13-smoke-pilot/artifact-index.json).

| Time (UTC) | Activity and hypothesis | Implementation / verification | Observation and decision | Reference |
| --- | --- | --- | --- | --- |
| Before 21:03, reconstructed | Inspect main and determine whether an actual Astra deployment is usable | `git status`, code/docs review, `codex --version`, `codex exec --help`, config metadata inspection | Clean base `8417fbb`; M3 runner and baselines already merged; config's default was `gpt-6-sol`, so explicit Astra selection was required | Git log; `docs/ASTRA_PILOT_STATUS.md` |
| About 21:03, reconstructed | Test `codex exec -m gpt-6-astra` as non-interactive adapter | Read-only ephemeral CLI nonce attempt | Failed before inference: local refresh token error and Azure tool-declaration error. No original CLI log archived. Switch to direct configured Responses endpoint | `docs/ASTRA_PILOT_STATUS.md` |
| About 21:04, reconstructed | Verify real Foundry Astra connectivity without assuming the interactive session proves it | One direct HTTPS Responses nonce request using configured endpoint and environment key | HTTP 200, model `gpt-6-astra`, response ID `resp_02fd1c4045f5ea08006af77b5754688195aaad69b1d470884b`; 18/11 tokens, 1.45849 s. Raw provider response not saved; model identity is supported by contemporaneous tool output but not independently replayable from a raw file | [Sanitized result](../../reports/experiments/2026-11-13-smoke-pilot/results.json) |
| 21:07:43 (file mtime) | Build a small corpus to check data plumbing without claiming chess strength | `dataset build --synthetic-games 30 --seed 42 --min-ply 8 --stride 6 --max-plies 90` | 22 train/2 dev/6 generated test games; 308/28 positions in train/dev. Test not OS sealed; do not evaluate it | Dataset manifest SHA `962352c3d586a5800f6a0ebbb303358fae6986d0f6e8cf57594ded0deeb91f78` |
| 21:07–21:13, reconstructed | Label with existing `python-chess` Stockfish wrapper | Several bounded oracle score/batch attempts; direct UCI handshake diagnostic | Wrapper stalled after startup in this sandbox's threaded event loop; interrupted attempts produced no label files. Direct subprocess UCI worked. Add smoke-only transport; leave original M1 oracle untouched | [Smoke labeler](../../src/owc/oracle/direct_uci.py), `docs/ASTRA_PILOT_STATUS.md` |
| 21:13:58 | Label 28 development positions with Stockfish 17.1 | `timeout 60 python scripts/label_smoke_uci.py ... --nodes 1000` | 28 labels published; 28,033 actual nodes; cp_tanh side-to-move | Dev label manifest SHA `98577c2c59c611ce78e63baaaeca0c17a8003182dec4967eaa3d5803577e314e` |
| 21:14:06 | Label 308 train positions with same settings | `timeout 120 python scripts/label_smoke_uci.py ... --nodes 1000` | 308 labels published; 308,348 actual nodes | Train label manifest SHA `b8f833fddecab2cbf02bd6b2cd011df8bf5f5145fdcf1d115bde41bfba4bf92b` |
| 21:14:29 (file mtime) | Check equal-architecture conventional baseline | Adam RawMLP seed 11, 100 distinct train labels, five epochs, batch 32, lr .001 | 20 optimizer updates, 500 label presentations, final dev MAE 0.704238772392273. No initial random-model MAE recorded | [Adam manifest](../../reports/experiments/2026-11-13-smoke-pilot/adam-seed11.manifest.json) |
| 21:14:41.905101 | Attempt E0 through genuine Astra adapter, expecting a blind patch | Seed 11, 120 s run limit, no teacher feedback, adapter `json_object`, 6,000 output-token ceiling | DNS denied before provider response at 1.1067 s; no patch returned, no accepted revision. Hash unchanged; stop E0 progression and do not start E5 | [E0 manifest](../../reports/experiments/2026-11-13-smoke-pilot/e0-attempt1.manifest.json) |
| 21:14:43 (file mtime) | Preserve E0 failure | Runner failure manifest, one rejection event, one adapter diagnostic, initial zero revision | Original diagnostic contains internal endpoint and stays local. Provider tokens and cost for this attempt are **missing**, not zero | [Artifact index](../../reports/experiments/2026-11-13-smoke-pilot/artifact-index.json) |
| After 21:14, reconstructed | Measure available comparators and validate implementation | `baselines compare`, `.venv/bin/ruff check .`, `.venv/bin/pytest -q`, `git diff --check` | Zero MAE 0.722732457937384; handcrafted 0.2583377147571239; Adam 0.704238772392273. 65 tests passed. Synthetic data prevent scientific comparison | [Results](../../reports/experiments/2026-11-13-smoke-pilot/results.json) |
| 21:25 onward, derived archival work | Make failure and baseline evidence reviewable without touching original artifacts | `reproduce_metrics.py`, `validate_artifacts.py`; compact manifests and checksummed index | 17 local original files verified; three generated test files declared unverified. One derived-index correction added the previously omitted test manifest; no original data changed | [Local inventory](LOCAL_ARTIFACT_INVENTORY.md), [archive](../../reports/experiments/2026-11-13-smoke-pilot/) |

## Decisions and unresolved items

The initial hypothesis remains untested: the only genuine Astra response was
the nonce, not a weight patch. A failed model invocation is counted as one
attempt and zero completed model calls; it is not counted as a rejected
weight patch. The zero-model error at E0 iteration 0 and after failure is
recomputed from original development labels and an unchanged model hash,
not an original per-round metric. The raw nonce response and failed CLI log
were not saved and cannot be reconstructed. No hidden test metric, E5 run,
multi-seed estimate, cost in dollars or Elo exists.
