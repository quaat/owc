# Open Weight Chess (OWC)

An experimental research project exploring **direct neural-network weight programming**
for chess position evaluation, with future Stockfish-guided feedback and a
conventional gradient-trained baseline.

This repository is **not yet a playable chess engine**. M0 provides a tested
legal-position encoder, a small zero-initialized PyTorch network, and a CLI to
inspect its outputs. All scores are 0 until a later milestone edits or trains
the network.

See [the development plan](docs/DEVELOPMENT_PLAN.md) for milestones M0–M5 and
[agent instructions](AGENTS.md) for implementation standards.

## Install (Python 3.12, CPU)

A virtual environment is recommended:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

The explicit CPU PyTorch installation avoids bringing in unnecessary CUDA packages
on Linux. Dependencies are version-pinned in `pyproject.toml`.

## Inspect a position

```bash
owc inspect
owc inspect --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --json
python -m owc inspect --json
```

The output shows the canonical FEN, legal move count, active encoder features,
network parameter count, and the static evaluation. Positive scores will mean
**better for the side to move** after the network has learned a useful function.

Invalid or impossible FEN positions are rejected rather than silently normalized.

## M1: Stockfish position oracle

Install Stockfish separately (e.g. from your system package manager). The
`STOCKFISH_PATH` variable identifies the engine; `stockfish` from PATH
is the fallback.

```bash
export STOCKFISH_PATH=/usr/games/stockfish
owc oracle score --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --nodes 10000 --json
printf '%s\n' "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" > /tmp/start.fen
owc oracle batch --input /tmp/start.fen --output /tmp/start.jsonl --nodes 10000
```

The `oracle batch` command produces new, non-overwritten JSONL and companion
`.manifest.json` files; a failed batch does not publish partial output.
Scores are from the **side-to-move** perspective and include raw centipawns
(or mate distance), engine ID, engine binary SHA-256, requested node budget,
PV and actual nodes when available. `cp_tanh` uses
`tanh(cp / 400)` by default, and `wdl_expectation` requires engine WDL
data. Terminal positions are rejected because chess rules determine their result.
An identical node budget is not a bitwise reproducibility guarantee across
Stockfish versions or hardware.

## M1: Build a position dataset

The first dataset pipeline groups positions by their **source game** before
assigning train, development and test partitions using a stable hash. It
deduplicates identical canonical board states (ignoring move counters) across
all partitions. It cannot eliminate every *near-neighbor* or opening-family
leak, so claims must be based on careful held-out testing.

Use user-provided PGN material only when you have the rights to analyze it:

```bash
owc dataset build --pgn games.pgn --source-license "authorized" \
  --dev-dir ./datasets/run-001 --heldout-dir /tmp/owc-sealed/run-001
```

For a deterministic **smoke test only**, synthetic random legal games can be
generated without a PGN source:

```bash
owc dataset build --synthetic-games 100 --seed 42 \
  --min-ply 8 --stride 4 --max-plies 80 \
  --dev-dir ./datasets/smoke --heldout-dir /tmp/owc-sealed/smoke
```

Development output has `train.fen`, `dev.fen`, associated `*.meta.jsonl`
records, and a manifest; held-out output has `test.fen` and its metadata.
Both directories must be new and separated; held-out output **must be
outside the working tree**. Access isolation is only real if the held-out
folder is additionally owned/ACL-protected under a **different OS user or
separate secure process** that the agent cannot read. The path check alone
is not a security boundary.

Feed development FEN files to the oracle:

```bash
owc oracle batch --input ./datasets/smoke/train.fen \
  --output ./datasets/smoke/train.labels.jsonl --nodes 10000
```

Do not give an editing agent access to held-out files, labels or benchmark
results. For scientifically useful experiments, replace random synthetic
games with licensed, representative human/engine game data, including
endgame and tactical position sources.

## Developer checks

```bash
ruff check .
pytest -q
python -m owc inspect --json
```

## Encoding contract: `owc.raw781.v1`

781 floating-point inputs in absolute board coordinates (`a1 = 0`, `h8 = 63`):

- `0:768`: twelve 64-square one-hot planes ordered white pawn, knight,
  bishop, rook, queen, king; then the same six black pieces.
- `768`: 1 for White to move, 0 for Black.
- `769:773`: castling rights, white king-side, white queen-side,
  black king-side, black queen-side.
- `773:781`: en-passant target **file** (`a` to `h`), one-hot only
  when at least one legal en-passant capture actually exists. Otherwise zero.
  Therefore different FEN en-passant fields may encode identically when no
  legal en-passant capture is available.

The zero-initialized `781 → 32 ReLU → 16 ReLU → 1 tanh` network has
**25,569 trainable parameters** and returns 0 on every valid position. Its
exact input semantics must be versioned when changed.

## Current boundaries

- `python-chess` supplies legal move generation, position validation, and
  outcome detection; the neural network only computes a **static score**.
- The M1 Stockfish oracle is available; datasets, agent patcher, supervised training,
  independent search and UCI server remain future milestones.
- Stockfish-derived labels and benchmark results must keep strict test-set
  isolation and explicit engine/version/settings provenance.
