# Issue #17: first Foundry Astra pilot

The adapter at `owc.experiments.astra_adapter` uses the configured Microsoft
Foundry `/openai/v1/responses` endpoint and an API key supplied only through
`AZURE_OPENAI_API_KEY`. The deployment is selected explicitly; an interactive
Codex session or a self-declared actor string is not connection evidence.

## Adapter configuration

Set `OWC_ASTRA_ENDPOINT` to the approved HTTPS inference base URL ending in
`/openai/v1`, `OWC_ASTRA_DEPLOYMENT` to the actual deployment, and
`OWC_ASTRA_ACTOR` to an identity such as `foundry:gpt-6-astra`. Set
`OWC_ASTRA_DIAGNOSTICS_DIR` to a dedicated, ignored directory owned by the
adapter process with mode 0700. It is required for a real call. The adapter
creates transcript and attestation files with mode 0600, refusing a
group/world-accessible directory. The transcript includes the exact provider
request and response bytes in base64, development positions, the internal
endpoint, hashes, IDs and usage; keep it under restricted access and never
commit it. The paired `.public.json` attestation omits positions and endpoint
but must still be reviewed before publication. A failed diagnostic write
causes the adapter to exit without emitting a patch, including after a
provider response.

The reported model must equal `OWC_ASTRA_DEPLOYMENT`. If Foundry reports an
underlying model identifier for a deployment alias, set
`OWC_ASTRA_ALLOWED_RESPONSE_MODEL` to that **one explicitly verified value**.
The restricted transcript records both names and the policy used. This
operator mapping is not independent proof of the provider's model identity.
The adapter reads one `owc.agent-run.v1` JSON request from stdin and writes
one `owc.weightpatch.v1` JSON object to stdout. It requests JSON object
format from Responses, records the provider response ID, model, request ID
header, usage and latency in a separate diagnostic JSON file, and never
persists the API key. The runner checks the actor and M2 validates all edits
before committing a revision. Failed edits leave the model unchanged and
write a rejection event.

The model receives the 781-feature contract, tensor shapes and indexing,
current nonzero weights (at most 512 cells), the current hash, blind probes,
and up to five compact preceding-round summaries. The complete UTF-8 request
has a 256,000-byte adapter limit and 4,096 bytes reserved as headroom. Oldest
summaries are omitted if needed; full patches remain in the local event log.
If even the reduced request is too large, the runner records `RequestSizeError`
before invoking the agent. Prior teacher feedback appears only after its
budgeted disclosure. Full development MAE after each patch is saved in
orchestration events, where the operator can inspect it, but is not disclosed
to the editing agent and is not an independent blind-test metric. A zero
model needs connected paths through all three linear layers to become
position-sensitive.

`CommandAgent` executes with the operator's OS permissions. Keep all sealed
test positions and labels under another OS account or a genuinely isolated
service. Dataset path names alone do not provide that separation. The adapter
does not execute model-supplied code. Do not put a provider key in a command
line, Git, a manifest, or diagnostics.

## Small development run

Use a licensed and attributed PGN for research claims. The following random
games are **integration smoke data only**:

```bash
owc dataset build --synthetic-games 30 --seed 42 --min-ply 8 \
  --stride 6 --max-plies 90 --dev-dir datasets/astra-smoke-30 \
  --heldout-dir /tmp/owc-astra-smoke-heldout-30
timeout 60 python scripts/label_smoke_uci.py \
  --input datasets/astra-smoke-30/dev.fen \
  --output datasets/astra-smoke-30/dev.labels.jsonl \
  --stockfish /path/to/stockfish --nodes 1000 --hash-mb 16
timeout 120 python scripts/label_smoke_uci.py \
  --input datasets/astra-smoke-30/train.fen \
  --output datasets/astra-smoke-30/train.labels.jsonl \
  --stockfish /path/to/stockfish --nodes 1000 --hash-mb 16

export OWC_ASTRA_ENDPOINT='https://YOUR-FOUNDRY-ENDPOINT/openai/v1'
export OWC_ASTRA_DEPLOYMENT='gpt-6-astra'
export OWC_ASTRA_ACTOR='foundry:gpt-6-astra'
export OWC_ASTRA_DIAGNOSTICS_DIR='runs/astra-smoke-e0/diagnostics'
owc experiment run --dev-labels datasets/astra-smoke-30/dev.labels.jsonl \
  --output-dir runs/astra-smoke-e0 --feedback-rounds 0 --seeds 11 \
  --max-seconds 180 --agent-id "$OWC_ASTRA_ACTOR" \
  --agent-command 'python -m owc.experiments.astra_adapter'
```

For E5, choose a fresh output directory and diagnostics directory, set
`--feedback-rounds 5 --probe-count 4 --max-label-disclosures 20`, and allow a
realistic wall-clock budget. E0 discloses zero labels. E5 discloses at most
20 precomputed development labels, counting repeats. Both runs use no live
Stockfish queries; the upstream label-generation nodes and position counts
must still be reported. Compare with `python -m owc.baselines train` and
`python -m owc.baselines compare` using the same corpus and encoding.

The direct-UCI labeler is a local smoke transport used when this sandbox
cannot run `python-chess`'s threaded UCI wrapper. Its manifest records
`transport=direct-uci-smoke-v1`. The normal `owc oracle batch` remains the
preferred M1 pipeline outside that restriction. Use the same transport,
Stockfish build, options and node budget for both train and dev splits.

Do not run E20/E50 until the E0/E5 adapter, patch behavior and costs are
reviewed, and their projected call/token budgets are confirmed. Independent
test claims require frozen checkpoints and an isolated evaluator process.
