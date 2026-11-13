# M3 protocol and deviations: issue #17

## Research question and decision rule

The planned question is whether a reasoning model can directly program a
useful static chess evaluator in the fixed OWC neural architecture, and how
its development error changes as bounded Stockfish feedback is disclosed.
The falsifiable first gate is a genuine model-generated patch that survives
M2 validation, creates connected nonzero paths, and produces different,
directionally sensible outputs for distinct material positions. A stronger
claim requires lower error than the zero model on an independent blind test,
uncertainty across at least three seeds, and comparison with the Adam and
handcrafted baselines under disclosed budgets.

## Planned conditions

The implemented runner defines E0 as one blind patch and E5/E20/E50 as one
blind patch plus 5/20/50 feedback rounds. Each feedback round discloses at
most `probe_count` precomputed development teacher labels and charges repeats.
E0 discloses none. The direct arm forbids gradients, backpropagation, finite
differences, random search and optimizer loops. Allowed M2 operations are
scalar `set`, scalar `add`, and `rank_one_add`, at most 128 operations and
10,000 addressed cells per patch. M2 validates names, indices, values and
current model hash before an immutable revision is committed.

Only **one failed E0 invocation** was actually attempted here. E5, E20 and
E50 were not run. The failed invocation did not receive a provider response,
so it is an integration attempt, not an E0 weight-programming result.

## Network and agent roles

Codex was the development agent editing Python and documentation. Microsoft
Foundry's `gpt-6-astra` deployment was verified once with a non-interactive
nonce request; that call did not propose weights. The experimental adapter
uses an HTTPS Responses request with explicit deployment, `json_object`
output format, a 6,000-output-token ceiling by default, and a 90-second
provider timeout by default. The E0 run had a 120-second total limit. The
adapter reads `owc.agent-run.v1` on stdin and would emit one
`owc.weightpatch.v1` object on stdout. Its [system instructions](../../src/owc/experiments/astra_adapter.py)
explain architecture, feature indices, edit syntax, the zero-network path
problem, and prohibited optimization. The [runner](../../src/owc/experiments/runner.py)
adds bounded nonzero state (512 cells), tensor summaries, blind probes, and
at most five preceding rounds. Prompt/source hashes at execution were not
saved; the archive records them as missing and gives a hash of the **current**
prompt separately.

The model never gets arbitrary filesystem tools. The command adapter itself
still inherits operator OS permissions; genuine test secrecy requires a
different OS principal or isolated benchmark service. No such boundary was
established in this smoke session.

## Evaluator and teacher

`RawMLP` is `owc.raw781.v1`: 781 inputs, 32 ReLU units, 16 ReLU units, one
tanh output, 25,569 float32 parameters. The first 768 features are 12
piece/square planes in absolute a1=0 to h8=63 order. Feature 768 is white
to move, 769–772 are castling rights, and 773–780 encode a legal en-passant
target file. Positive output favors the side to move. Direct sessions start
at all zeros. Adam's conventional baseline starts from seeded nonzero
Kaiming-style weights and trains with MSE/Adam; those gradients are strictly
outside the direct arm. The handcrafted baseline uses declared material
values and piece placement.

Stockfish 17.1 from Debian `17-1` was the teacher, binary SHA-256
`38faa5883b03652f847a87ed168b1bfee81b361db9584dae59a51cb91e69d9d6`.
Settings were Threads=1, Hash=16 MiB, 1,000 requested nodes per position,
UCI_ShowWDL=false and target `tanh(cp_stm/400)` (mate mapped to ±1).
The upstream teacher cost is 308,348 actual train nodes plus 28,033 dev
nodes. `live_stockfish_queries=0` in the experiment runner does not erase
that generation cost.

## Corpus, sampling and evaluation

For this integration smoke only, OWC generated 30 synthetic random legal
games with seed 42, minimum ply 8, stride 6, maximum ply 90. Its M1
source-game hash split assigned 22 games/308 positions to train,
2 games/28 positions to dev, and 6 games to an unsealed generated test
directory. The test contents were not inspected or scored. Exact FEN
deduplication applied; near-neighbor and opening-family leakage were not
excluded. The dev split has 4 opening and 24 middlegame positions by the
post hoc phase rule used in the field report, and no endgames. Dataset and
label hashes are in the [archive manifests](../../reports/experiments/2026-11-13-smoke-pilot/).

The primary measured smoke metric is mean absolute error against the
development Stockfish target. No confidence interval, ranking statistic,
playing strength or Elo is justified by this sample. Distinct training
labels, repeated label presentations, optimizer updates, agent calls,
tokens, node counts and wall time are separate budgets.

## Actual deviations from the planned research study

1. Representative licensed PGNs were not available with verified rights.
   Synthetic games were used solely to verify integration. No scientific
   performance conclusion should use them.
2. The sandbox's asyncio cross-thread wakeup stalled the normal
   `python-chess` UCI wrapper. A bounded direct-UCI smoke labeler produced
   the same M1 `cp_tanh` record shape and recorded its alternate transport.
3. `codex exec -m gpt-6-astra` failed before inference because of local auth
   and Azure tool-declaration errors. A direct Responses nonce call proved
   model connectivity, but its raw response was not durably archived.
4. The E0 adapter invocation failed at DNS resolution in the runtime
   sandbox. There was no model response, patch, or new evaluator revision.
5. The generated test directory shared the operator's OS permissions and
   was never a sealed benchmark. No test metric exists.
6. Only one supervised seed was trained. It used 100 distinct labeled train
   positions, 500 label presentations, five epochs and 20 Adam updates.
   Its initial random-model MAE was not recorded.

## Claim acceptance

Engineering connection: supported by the observed nonce response, with the
raw-provider-transcript archival limitation. Offline adapter and runner
contracts: supported by tests. Genuine Astra weight programming: **not
demonstrated**. Development improvement by Astra: **not measured**.
Independent generalization: **not measured**. Chess playing strength is out
of scope until M4 supplies an independent search/UCI engine.

A fresh model run may produce a different patch even with the same request.
Deterministic replay applies only to saved responses, patches and checkpoints;
none was created by Astra in this attempt.
