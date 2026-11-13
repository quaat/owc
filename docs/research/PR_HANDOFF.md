# Draft pull request: research(m3): archive Astra integration attempt and findings

**Linked issue:** #17. **Target:** `main`. **Source:**
`experiment/astra-m3-pilot`. **Merge:** leave for independent review.
This preserves the prepared PR body. The live PR records its own CI status.

## Summary

Preserves the first Foundry `gpt-6-astra` integration attempt and reports its
actual limit: a successful independent nonce connection, followed by an E0
weight-patch invocation that failed at DNS before any provider response.
No Astra-generated tensor edit, E5 feedback run, or independent benchmark
is claimed. The PR adds a narrow Responses adapter, bounded agent context,
failure auditing, a smoke-only direct-UCI label transport, exact baseline
budget accounting, and a reproducible compact research archive.

## What ran

30 synthetic random legal games gave 308 train and 28 dev positions.
Stockfish 17.1, 1,000 requested nodes per position, searched 308,348 train
and 28,033 dev nodes. The zero network, handcrafted evaluator and Adam
RawMLP had dev MAE 0.722732457937384, 0.2583377147571239 and
0.704238772392273 respectively. Adam used seed 11, 100 distinct train
labels, 500 presentations, 20 updates and five epochs. E0 seed 11 disclosed
zero labels; its one adapter invocation failed at DNS, returned no patch,
and left hash `11eb14dd2b0d22f7a1d3587f2469d0983159c739f83df90fb9448bebfb279f3c`
unchanged. E5/E20/E50 were not run. These synthetic data are integration
smoke evidence only.

## Validation

```bash
.venv/bin/ruff check .                     # passed
.venv/bin/pytest -q                        # 65 passed
.venv/bin/python scripts/research/reproduce_metrics.py --check
# verified six derived report files
.venv/bin/python scripts/research/validate_artifacts.py
# validated 20 local original files, three declared unverified test files,
# and six deterministic compact report products
git diff --check                            # passed
git diff --cached --check                   # passed
```

The live PR records GitHub Actions status and the exact workflow URL.

## Reproducibility and artifacts

Read `docs/research/M3_FIELD_TRIAL_REPORT.md`,
`docs/research/EXPERIMENT_PROTOCOL.md`,
`docs/research/RESEARCH_LOG.md`,
`docs/research/ARTICLE_EVIDENCE.md`, and
`docs/research/NEXT_STEPS.md`. Machine-readable summaries and a checksummed
index are in `reports/experiments/2026-11-13-smoke-pilot/`.
`scripts/research/reproduce_metrics.py --check` and
`scripts/research/validate_artifacts.py` replay locally while original
ignored artifacts are present. Raw synthetic corpus, labels, weights and
diagnostic remain local and are **not durably archived**. The raw diagnostic
contains an internal endpoint and is not published; original nonce provider
JSON was never saved. A fresh checkout can review the compact findings but
cannot independently recompute source labels or checkpoint metrics without
recreating the data and environment.

## Risks and scientific limits

No genuine Astra weight patch has been demonstrated. The 28-position dev
split has no endgames, there was one Adam seed, budgets are unmatched, and
test data were not protected by a different OS principal. The normal
`python-chess` threaded UCI wrapper stalled in this sandbox; a direct-UCI
smoke transport was recorded as a deviation. No test metric, interval,
playing strength or Elo exists. Prompt/source hashes at E0 execution were
not captured; current prompt hash is explicitly distinguished.

## Reviewer questions and next gate

1. Is the adapter's input contract and diagnostics boundary suitable for
   one network-permitted call in a truly test-isolated process?
2. Which licensed corpus and durable artifact store should support the
   first scientific E0/E5 trial?
3. Should the smoke-only direct-UCI transport remain a separate fallback
   or be removed after the environment issue is fixed?

Next issue gate: obtain one real model-generated, M2-accepted patch with
position-sensitive probes and complete provider/patch/checkpoint evidence;
then attempt bounded E5. Review projected costs before E20/E50. Do not
merge this as evidence of Astra chess success.
