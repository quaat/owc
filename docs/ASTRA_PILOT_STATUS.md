# Issue #17 status: Foundry connection and development smoke run

Date: 2026-11-13. Branch: `experiment/astra-m3-pilot`. Base commit:
`8417fbb939dc465aca269aef80dcd7cad5bc940e` (`origin/main` matched).
This status is ready to post to issue #17 after repository/network access is
available. No experiment success or held-out result is claimed.

## Model connection evidence

One independent, non-interactive HTTPS Responses request reached the locally
configured Microsoft Foundry inference endpoint and returned HTTP 200,
`status=completed`, `model=gpt-6-astra`, and the requested nonce. Response ID:
`resp_02fd1c4045f5ea08006af77b5754688195aaad69b1d470884b`.
Usage: 18 input tokens, 11 output tokens, 29 total; latency 1.46 seconds.
The API key came from `AZURE_OPENAI_API_KEY` and was never stored in the
response artifact. This proves model access only. It is **not** a weight
patch or chess experiment.

The local `codex exec -m gpt-6-astra` path failed before inference: Codex
reported an expired local refresh token and the Azure service rejected its
tool declaration because an image-generation deployment header was absent.
The dedicated Responses adapter therefore uses the same configured inference
base URL and API key directly. Its JSON-object response mode has only offline
fixture validation so far; a real structured patch call has not completed.

## Development data and baselines

30 synthetic random legal games built by OWC seed 42, stride 6, plies 8..90.
This source has no external license requirement, but is unsuitable for
scientific chess-performance conclusions. Group split: 22 train games,
2 development games, 6 test games. The latter are **not sealed by OS access**
and were neither labeled nor evaluated here. The development split has
28 positions: 4 opening, 24 middlegame, 0 endgame; 23 show a material
imbalance. Training has 308 positions. Development FEN SHA-256:
`f3bbd31834e2e131af44a0733f8316a13085f956e17904fc7384afcd8cc6ecab`.
Development labels SHA-256:
`0163e99d8436290e0edff3dbcdcfed7e54f35d6aa35de4d6a96c38c80b899b03`.
Training labels SHA-256:
`c13299cd1b815f085832f62f461ce06c3e259422ac53e153eaf4cb1f4f7f29a1`.

Stockfish 17.1 Debian package `17-1`, binary SHA-256
`38faa5883b03652f847a87ed168b1bfee81b361db9584dae59a51cb91e69d9d6`,
Threads 1, Hash 16 MiB, requested 1,000 nodes per position,
`cp_tanh=tanh(cp_stm/400)`. The direct-UCI smoke transport searched
308,348 train nodes and 28,033 development nodes in total. These are
upstream teacher-generation costs even though the direct-edit runner makes
zero live Stockfish queries.

| Arm | Development MAE (28 synthetic positions) | Exposures and compute |
| --- | ---: | --- |
| Initial zero network | 0.722732 | 0 training labels; 25,569 parameters |
| Handcrafted material/placement | 0.258338 | Fixed chess rules; 0 training labels |
| Adam RawMLP, seed 11 | 0.704239 | 100 distinct train labels, 500 presentations, 20 optimizer updates, 5 epochs, 0.50 s training |
| Astra E0 | unavailable | 0 teacher disclosures; patch call blocked before inference |
| Astra E5 | not started | Requires E0 patch and accepted behavior |

Adam checkpoint hash:
`2d2b4c805af32abb31cc210bbbaf085aa4d6f2ca39025ccceeeb3e662c3dbb4a`.
The handcrafted rule has explicit prior chess knowledge. The supervised arm
uses the same `owc.raw781.v1` encoding and 781→32→16→1 architecture but
starts from nonzero Kaiming-style weights. Its poor smoke MAE is not a
generalization result.

## E0 attempt and acceptance state

The real adapter was invoked through `owc experiment run` at seed 11 with a
120-second wall budget. The operating-system sandbox denied DNS resolution
for the configured Foundry endpoint. The run is marked failed and stores
the full development-only request and a rejection event under
`runs/astra-smoke-e0-attempt1/`. No teacher target was present in the E0
request. Model hash before and after rejection is identical:
`11eb14dd2b0d22f7a1d3587f2469d0983159c739f83df90fb9448bebfb279f3c`.
There was no model response, accepted patch, revision, or position-sensitive
behavior. E5 and E20/E50 were not attempted.

The independent held-out gate is blocked: the generated test split shares
the same OS owner and is therefore accessible to the editing process.
A licensed representative corpus with endgames and tactics is also missing.

## Validation and environment

Python 3.12.15, PyTorch 2.5.1+cpu, numpy 1.26.4, python-chess 1.999,
pytest 8.3.4, Ruff 0.9.6. Host reports 16 logical CPUs and 31 GiB RAM.
The normal `python-chess` threaded Stockfish wrapper hangs after startup in
this sandbox; the bounded direct-UCI smoke labeler completed both splits.

Commands run successfully after the changes:

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
timeout 60 .venv/bin/python scripts/label_smoke_uci.py --input datasets/astra-smoke-30/dev.fen --output datasets/astra-smoke-30/dev.labels.jsonl --stockfish /tmp/owc-stockfish/usr/games/stockfish --nodes 1000 --hash-mb 16
timeout 120 .venv/bin/python scripts/label_smoke_uci.py --input datasets/astra-smoke-30/train.fen --output datasets/astra-smoke-30/train.labels.jsonl --stockfish /tmp/owc-stockfish/usr/games/stockfish --nodes 1000 --hash-mb 16
.venv/bin/python -m owc.baselines train --train-labels datasets/astra-smoke-30/train.labels.jsonl --dev-labels datasets/astra-smoke-30/dev.labels.jsonl --output-dir runs/astra-smoke-supervised-seed11 --seed 11 --epochs 5 --batch-size 32 --learning-rate 0.001 --max-train-examples 100
.venv/bin/python -m owc.baselines compare --dev-labels datasets/astra-smoke-30/dev.labels.jsonl --supervised-checkpoint runs/astra-smoke-supervised-seed11
```

Next gate: run one real structured Astra patch call in a network-permitted,
test-isolated process, verify a nonconstant evaluator and replay its audit
checkpoint, then run bounded E5. Review actual call/token costs before any
E20/E50 expansion. Do not merge or claim independent evidence yet.
