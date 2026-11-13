# M3 field trial report: first Foundry Astra integration attempt

**Scope:** development-only engineering smoke, 2026-11-13, OWC issue #17.
The [structured results](../../reports/experiments/2026-11-13-smoke-pilot/results.json),
[run manifests](../../reports/experiments/2026-11-13-smoke-pilot/),
[trajectory](../../reports/experiments/2026-11-13-smoke-pilot/trajectory.csv),
and [artifact index](../../reports/experiments/2026-11-13-smoke-pilot/artifact-index.json)
are the numerical sources for this report. They were derived from unchanged
ignored local artifacts and can be checked with
`python scripts/research/validate_artifacts.py` while those originals exist.

## 1. Executive summary

A non-interactive request to the configured Microsoft Foundry Responses
endpoint returned a `gpt-6-astra` nonce response. The first E0 weight-edit
attempt then failed at sandbox DNS resolution before receiving any model
response. Consequently, **Astra did not program OWC weights in this trial**.
No direct-edit performance, position sensitivity, or improvement was
measured. Local data plumbing, the adapter's offline contract, rejection
auditing, Stockfish labeling and conventional baselines were exercised.
The 30-game corpus was synthetic random play, so its error numbers are
integration diagnostics and no chess-strength evidence.

## 2. Research questions

The intended hypothesis was that explicit model-authored tensor edits could
construct a position-sensitive neural evaluator from a zero network and,
with E5 feedback, reduce development Stockfish error. The alternative
comparators were a zero network, a fixed material/placement heuristic and
Adam training in the same architecture. This trial did **not** test the
direct-edit hypothesis because no Astra patch was returned.

## 3. Experimental setup and roles

Codex edited source and orchestrated the work. The deployment named
`gpt-6-astra` was the proposed experimental weight editor; Stockfish 17.1
was the reference oracle; `RawMLP` was OWC's neural evaluator. Adam's
gradient updates belonged only to the conventional baseline. The fixed
heuristic embodied predefined chess knowledge.

The host reported 16 logical CPUs and 31 GiB RAM. Python 3.12.15 ran
PyTorch 2.5.1+cpu, numpy 1.26.4 and python-chess 1.999. Source base was
`8417fbb939dc465aca269aef80dcd7cad5bc940e`, with a dirty working tree
at experiment time. Exact executable/source commit for the adapter was not
recorded because the branch was uncommitted; the archive does not pretend
the base commit contains the new adapter.

## 4. Dataset and Stockfish oracle

OWC generated 30 synthetic legal games at seed 42. A source-game hash split
made 308 train positions from 22 games and 28 dev positions from two games.
Six test games were generated into a same-owner `/tmp` directory, which was
not protected and was never evaluated here. The dev set contained four
opening and 24 middlegame positions and no endgames. Twenty-three dev
positions had a material imbalance. These phase counts were classified
afterward; they were not stratification targets. Synthetic random games are
not representative of chess practice or tactics.

Stockfish 17.1 Debian `17-1` binary SHA-256 was
`38faa5883b03652f847a87ed168b1bfee81b361db9584dae59a51cb91e69d9d6`.
The run used one thread, 16 MiB hash, no UCI WDL output and 1,000 requested
nodes per position. Side-to-move centipawns became `tanh(cp/400)`; mates
became ±1. The local smoke labeler used direct UCI after the normal
`python-chess` threaded wrapper stalled. The alternate transport is in
the original label manifests. Actual searched nodes totaled 308,348 for
train and 28,033 for dev; those generation costs precede any experiment
feedback. The dev labels SHA-256 is
`0163e99d8436290e0edff3dbcdcfed7e54f35d6aa35de4d6a96c38c80b899b03`.
No independent hidden-test labels exist.

## 5. Astra weight-programming mechanism

The proposed editor receives the `owc.raw781.v1` feature map and
781→32→16→1 ReLU/tanh network specification, named tensor shapes, current
hash, bounded nonzero weights, blind development probes, and permitted
M2 operations. It may return scalar `set`/`add` and structured rank-one
additions, each subject to M2 hash, shape, range and finiteness validation.
The zero network requires connected nonzero paths through all three linear
layers; changing only the output bias would fail the behavioral goal.
Gradient methods and black-box numerical search are prohibited in this arm.
The [protocol](EXPERIMENT_PROTOCOL.md) records the full disclosure policy
and the actual deviations.

The independent nonce request returned HTTP 200, model `gpt-6-astra`,
response ID `resp_02fd1c4045f5ea08006af77b5754688195aaad69b1d470884b`,
18 input tokens, 11 output tokens and 1.4584902620408684 seconds. Its raw
provider JSON was not saved, reducing later independent verification. The
adapter's `json_object` weight-patch mode was tested only with a fixture;
there was no provider response in the E0 attempt.

## 6. Execution and budgets

E0 at seed 11 started `2026-11-13T21:14:41.905101+00:00` with a 120-second
run limit. It selected four blind probes, disclosed zero teacher targets,
and attempted one provider call. DNS failed in 0.00776629289612174
adapter seconds; the run recorded 1.106812443002127 total seconds. Completed
model calls: 0. Accepted patches: 0. Rejected patches: 0; no patch was
returned. Failed invocations: 1. Input/output tokens and monetary charge
for this attempt are **unavailable**, not recorded as zero. No E5, E20 or
E50 call followed.

The supervised baseline used seed 11, 100 distinct labeled train positions,
five epochs, batch size 32, Adam learning rate 0.001 and 20 updates. It
presented those labels 500 times, evaluated all 28 dev positions after
each epoch, and recorded 0.49710616702213883 training seconds. Agent
rounds, optimizer updates, labeled positions, tokens and engine nodes are
different units; this trial did not match them into a fairness budget.

## 7. Results and trajectory

Development MAE values below are from [results.json](../../reports/experiments/2026-11-13-smoke-pilot/results.json).
The zero-model value was recomputed from original labels. Adam and
handcrafted values were recomputed through the existing comparison code.

| Evaluator | Dev MAE, 28 synthetic positions | Status |
| --- | ---: | --- |
| Zero RawMLP | 0.722732457937384 | Deterministic reference |
| Handcrafted material/placement | 0.2583377147571239 | Fixed prior chess rules |
| Adam RawMLP, seed 11 | 0.704238772392273 | Completed, same encoding and architecture |
| Astra E0 | unavailable | Failed before inference; no edited evaluator |
| Astra E5 | unavailable | Not run |

Adam epoch dev MAE was 0.7227121591567993,
0.7162044644355774, 0.7110034823417664,
0.7055872678756714, then 0.704238772392273. The original trainer did
not record the initial Kaiming-model error or intermediate checkpoint hashes.
The E0 initial and final revision are both
`5fdedee893f44c5aa1e0c9bc5a27bec9`, hash
`11eb14dd2b0d22f7a1d3587f2469d0983159c739f83df90fb9448bebfb279f3c`.
The failed invocation left the zero model unchanged. No material-direction
or position-sensitivity result can be attributed to Astra.

No graph of direct-edit error versus rounds or cost was produced because
there is no accepted patch trajectory. A chart comparing three methods on
28 synthetic positions would invite an unsupported performance inference;
the numerical table retains the engineering evidence without that effect.

## 8. Baseline comparison

The handcrafted baseline's lower MAE on this tiny synthetic dev set reflects
its explicit material and placement prior; it does not establish broad
superiority. Adam's final MAE is slightly below the zero network's on this
split, but one seed, 100 sampled train labels and no endgames cannot
establish a robust learning curve. No direct-edit checkpoint exists to
compare. Adam checkpoint model hash is
`2d2b4c805af32abb31cc210bbbaf085aa4d6f2ca39025ccceeeb3e662c3dbb4a`.

## 9. Failure analysis

The `codex exec` route failed before inference from local refresh-token
and Azure tool-declaration errors. The direct Responses nonce call then
worked, but the full experiment command ran inside a network-restricted
sandbox and could not resolve the Foundry host. The runner saved its full
development-only request, marked the event rejected, and retained the
current checkpoint. Raw diagnostic JSON remains local because it contains
the internal endpoint. Its checksum appears in the artifact index.

The normal Stockfish wrapper also stalled in this sandbox. Direct UCI
completed 336 labels at the same declared settings. This is a labeling
transport deviation, not a new chess evaluator or model experiment.

## 10. Threats to validity

The three largest threats are: no genuine Astra patch or development
measurement; synthetic unrepresentative positions with no endgames; and
no OS-isolated held-out benchmark. Additional limits include one Adam seed,
no matched token/label/time budget, unarchived raw nonce response,
uncommitted adapter source at execution, and Stockfish's platform-dependent
search behavior even at fixed node limits. Model self-declared actor strings
alone are not authentication.

## 11. Reproducibility and archival status

With the original ignored local artifacts still present, run:

```bash
.venv/bin/python scripts/research/reproduce_metrics.py --check
.venv/bin/python scripts/research/validate_artifacts.py
.venv/bin/ruff check .
.venv/bin/pytest -q
```

The validator checks 20 local original file hashes, M1 label/source
manifests, checkpoint load/hash, E0 unchanged state and generated summaries.
It does not open the generated test split. The public archive has compact
results and hashes, but no durable raw-artifact store; a fresh checkout
cannot regenerate the labels or supervised metrics without rebuilding the
synthetic corpus and Stockfish labels. A fresh Astra invocation could return
different weight edits even if its request were copied exactly.

## 12. Conclusions

The only confirmed model behavior is nonce generation by the configured
Foundry deployment. The weight-editing hypothesis is still untested. The
failure audit, label provenance, supervised accounting and local replay
provide an engineering basis for a properly isolated first trial. There is
no independent evidence and no chess playing-strength claim.

## 13. Recommended next experiment

First run one structured patch request through the real adapter in a
network-permitted process **that cannot read the held-out benchmark**.
Archive the raw provider response, exact prompt/source hashes, usage,
latency, M2 patch, initial/final checkpoints and behavioral position probes.
Only if that gate passes should a bounded E5 trial be attempted on licensed,
representative development data, followed by frozen-model independent
evaluation and multi-seed replication. E20/E50 need reviewed call and
compute budgets before launch.
