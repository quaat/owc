# M3: Supervised and handcrafted chess evaluation baselines

This is a **separate experimental arm** from M2 direct tensor patching.

## A. Handcrafted baseline

`python -m owc.baselines handcrafted --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"`

The rule uses fixed material values (P=100, N=320, B=330, R=500,
Q=900 centipawns) and disclosed simple piece-placement bonuses.
It converts the side-to-move centipawn score to `tanh(cp / 400)`.
It has **no** tree search and is not comparable to Stockfish Elo.

## B. Adam-trained neural baseline

Train **exactly the same model and input encoding as M2**:
`RawMLP` 781 → 32 → 16 → 1 (25,569 parameters, `owc.raw781.v1`).
Unlike weight programming, **this arm explicitly uses gradients**.
The model is **randomly initialized with nonzero Kaiming-style weights**
to break the symmetry of M2's zero-initialized network. It uses PyTorch
Adam/MSE on Stockfish-labeled train positions, and tracks train/dev MAE.

Prepare M1 source-group-split corpora and Stockfish labels:

```bash
owc oracle batch --input datasets/run-001/train.fen \
  --output datasets/run-001/train.labels.jsonl --nodes 10000
owc oracle batch --input datasets/run-001/dev.fen \
  --output datasets/run-001/dev.labels.jsonl --nodes 10000
```

Train and evaluate:

```bash
python -m owc.baselines train \
  --train-labels datasets/run-001/train.labels.jsonl \
  --dev-labels datasets/run-001/dev.labels.jsonl \
  --output-dir runs/supervised-seed11 --seed 11 --epochs 10 \
  --batch-size 32 --learning-rate 0.001

python -m owc.baselines compare \
  --dev-labels datasets/run-001/dev.labels.jsonl \
  --supervised-checkpoint runs/supervised-seed11 \
  --direct-session runs/experiment-001/E0-seed-11/weights
```

Use `--max-train-examples` to bound distinct teacher examples during
training for budget comparisons. This cap does **not** equate gradient
epochs/updates with agent feedback rounds, nor account for upstream
Stockfish labeling expense. Report both teacher-label generation cost
and training resources when making claims.

Results are per-DEV position MAE (handcrafted/supervised/direct).
The `compare` command deliberately refuses the TEST split. To establish
held-out model quality, run an independent evaluation service under
separate OS credentials, after checkpoints are frozen and selection is
finished. Neither the CLI nor filename checks constitute a security
boundary against an unrestricted user process.

Current CI uses synthetic randomly played games and tiny node budgets
**for protocol validation only**, never chess-strength evidence.
Models are stored as float32 NumPy archives with pickle disabled,
SHA-256 bytes checks and verified model-content hashes.
