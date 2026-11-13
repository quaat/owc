# Draft issue #17 update

This is the prepared issue comment text. The live issue comment records the
PR URL, commit and GitHub Actions status.

The first Foundry connection check reached `gpt-6-astra` non-interactively
(HTTP 200, response ID
`resp_02fd1c4045f5ea08006af77b5754688195aaad69b1d470884b`, 18 input
and 11 output tokens). The raw provider response was not saved. This is
connection evidence only, not a tensor-edit result.

The first E0 seed-11 weight-edit attempt failed before provider inference
because the runtime sandbox could not resolve the endpoint. It disclosed
zero teacher labels, returned no patch and left the initial zero-model
hash unchanged. E5/E20/E50 have not run. The adapter, bounded context,
failure audit, smoke labeler and validation tests are prepared locally on
`experiment/astra-m3-pilot`.

The 30-game synthetic integration corpus had 308 train/28 dev positions.
Stockfish 17.1 at 1,000 requested nodes searched 336,381 nodes across
both splits. On the 28-position synthetic dev split, MAE was 0.722732457937384
for the zero model, 0.2583377147571239 for the handcrafted heuristic, and
0.704238772392273 for Adam RawMLP seed 11. Adam used 100 distinct labeled
train positions, 500 presentations and 20 updates. These numbers do not
support chess-strength or direct-programming claims. No test evaluation
was conducted; OS-level isolation was absent.

The review archive is in `docs/research/` and
`reports/experiments/2026-11-13-smoke-pilot/`. Local validation passed:
Ruff, 65 pytest tests and a hash/metric replay of 20 original local files.
The raw labels, checkpoint and diagnostic are not in durable storage;
the archive flags this limitation explicitly. The live PR records CI status.

Next review decision: provide a network-permitted process that cannot read
the sealed test service, choose a licensed representative corpus and an
artifact retention location, then run one fully archived structured Astra
patch call. Require a valid M2 revision and position-dependent behavior
before E5. Budget E20/E50 separately.
