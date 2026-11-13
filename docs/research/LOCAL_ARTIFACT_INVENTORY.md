# Local artifact inventory before research archival

Inventory taken 2026-11-13 from `experiment/astra-m3-pilot` at base
`8417fbb939dc465aca269aef80dcd7cad5bc940e`. Originals were not moved,
replaced, or relabeled. All `datasets/` and `runs/` files are ignored by Git.
The machine-readable [artifact index](../../reports/experiments/2026-11-13-smoke-pilot/artifact-index.json)
records every local file's size, SHA-256, access and retention status.

## Working-tree source and notes

Modified: `src/owc/baselines/supervised.py`,
`src/owc/experiments/runner.py`, `tests/test_baselines.py`, and
`tests/test_experiments.py`.

Added: `src/owc/experiments/astra_adapter.py`,
`src/owc/oracle/direct_uci.py`, `scripts/label_smoke_uci.py`,
`tests/test_astra_adapter.py`, `tests/test_smoke_uci.py`,
`docs/ASTRA_PILOT.md`, and `docs/ASTRA_PILOT_STATUS.md`.
These files constitute the local implementation and first-session report;
they were uncommitted when this inventory was taken.

## Original data and run artifacts

| Group | Original files | Status |
| --- | --- | --- |
| Dataset | `datasets/astra-smoke-30/manifest.json`, `train.fen`, `dev.fen`, `train.meta.jsonl`, `dev.meta.jsonl` | Local synthetic corpus; group split 22 train/2 dev/6 test games |
| Teacher | `train.labels.jsonl`, `dev.labels.jsonl` and each `.manifest.json` in the same dataset directory | Local Stockfish 17.1 labels; 308 train and 28 dev; no test labels |
| E0 attempt | `runs/astra-smoke-e0-attempt1/E0-seed-11/manifest.json`, `events.jsonl`, `weights/HEAD`, and the one initial revision's `manifest.json` and `weights.npz` | Failed before inference; no accepted patch |
| Adapter diagnostic | One JSON file in `runs/astra-smoke-e0-attempt1/diagnostics/` | Local only; contains an internal endpoint and request FENs; do not publish raw |
| Supervised | `runs/astra-smoke-supervised-seed11/manifest.json`, `weights.npz` | Completed Adam checkpoint, seed 11; local only |
| Generated test split | `/tmp/owc-astra-smoke-heldout-30/manifest.json`, `test.fen`, `test.meta.jsonl` | **Not sealed by OS permissions.** Paths and sizes inventoried; contents were not inspected, labeled or evaluated for this handoff |
| Connection script | `/tmp/owc_astra_connection.py` | Original nonce script; contains internal endpoint and remains local. SHA-256 `964a64a01a3712904988b8fbf421fcfebf91b42bc715cf131fcc960006d761c` |
| Stockfish package and executable | `/tmp/stockfish_17-1_amd64.deb`, `/tmp/owc-stockfish/usr/games/stockfish` | Local 61 MB package and 78 MB extracted executable; hashes in artifact index; neither committed |

The dataset directory has nine files; the E0 attempt has six files including
its diagnostic; the supervised run has two. The generated test split has three
files, plus three external local inputs listed above. Source PGNs found elsewhere on the machine had no verified license and
were not used. Stockfish's local extracted binary is
`/tmp/owc-stockfish/usr/games/stockfish`; it is not a repository artifact.

## Evidence that was never saved as an original file

The independent non-interactive Foundry nonce response was observed in tool
output, including response ID and token usage, but its raw response was not
written to a durable file. The compact connection summary in
`docs/ASTRA_PILOT_STATUS.md` is a subsequent note, not the original provider
transcript. The failed `codex exec` diagnostics likewise have no retained
original log file. No original E5, E20, E50, accepted patch, or blind-test
artifact exists.

## Derived archive boundary

Files under `reports/` and `docs/research/` created after this inventory are
derived summaries or validation code. They do not replace any original run
record. No durable external artifact store has been approved; local ignored
artifacts may be lost if the workspace is removed.
