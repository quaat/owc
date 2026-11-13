# 2026-11-13 synthetic development smoke archive

Permanent archive ID: `owc-m3-20261113-smoke-pilot`.
This is a failed Astra E0 **integration attempt** plus a completed Adam
baseline and two fixed comparator evaluations. It is not a successful
weight-programming trial or an independent benchmark.

| File | Contents |
| --- | --- |
| [results.json](results.json) | Exact aggregate development metrics and connection observation |
| [e0-attempt1.manifest.json](e0-attempt1.manifest.json) | Failed E0 provenance, zero disclosures and unchanged model |
| [adam-seed11.manifest.json](adam-seed11.manifest.json) | Supervised settings, exposure/update counts and epoch values |
| [trajectory.csv](trajectory.csv) | Per-iteration/epoch metrics; blank cells mean unrecorded |
| [metrics.csv](metrics.csv) | Compact comparator MAE table; Astra value intentionally blank |
| [artifact-index.json](artifact-index.json) | Original local file hashes, sizes, access and retention status |

The raw 30-game synthetic corpus, Stockfish labels, run diagnostics and
checkpoint files remain ignored under `datasets/` and `runs/`. They are
**not** archived durably by this repository. The raw adapter diagnostic
contains an internal inference endpoint and is not published. Generated
test positions are not copied, inspected or evaluated here. The index's
test hashes come only from the original dataset manifest and were not
independently rehashed.

Recompute the compact files while originals are available:

```bash
.venv/bin/python scripts/research/reproduce_metrics.py --check
.venv/bin/python scripts/research/validate_artifacts.py
```

The validator confirmed 20 local original files (including the connection
script and Stockfish package/executable) and three declared,
unverified generated-test files on 2026-11-13. A fresh checkout without
the ignored originals can inspect the self-contained numerical summaries,
but cannot independently replay their source labels or checkpoint. The
raw nonce response from the successful Foundry connection was never saved;
its compact observation is explicitly archival-limited.

See the [standalone report](../../../docs/research/M3_FIELD_TRIAL_REPORT.md)
for methods, results, limitations and the next gate.
