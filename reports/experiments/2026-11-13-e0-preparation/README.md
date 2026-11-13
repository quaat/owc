# Issue #20: isolated E0 preparation, no inference call

The [preparation record](preparation.json) is a byte-for-byte copy of the
private local `runs/astra-e0-preflight-clean-20261113/preparation.json`
(SHA-256 `edf13f1fa505cccb7ef1390da3a49bdc42674c4fe5e881250b5a203576198346`).
It was produced from clean commit `5a5d08d591c681eb723516ebe9fa0d558c30eb50`
using `--probe-only`. No provider inference, model-generated patch, E5 run,
Stockfish labeling or held-out evaluation occurred.

The Bubblewrap process used a read-only source mount, empty home and `/tmp`,
masked datasets and runs, and a private writable diagnostics directory. It
reported that the generated test path was inaccessible. The access probe
checked path visibility without reading test contents. An authenticated
`GET /models` from the same namespace returned HTTP 200 and listed the
requested deployment. This supports network and credential reachability;
it does not verify Responses `json_object` behavior or the actual model
identity of a future patch response.

The existing test split is synthetic and was not an independently sealed
blind benchmark. The run plan permits exactly one E0 call, with zero teacher
disclosures, at most 3,000 output tokens, a 256,000-byte request cap and a
150-second total wall limit. The provider model listing supplied no price,
so no defensible dollar ceiling is available. The billable call remains
unstarted while that cost gate is reviewed. No E5 or larger experiment is
authorized by this preparation result.

The private preflight and offline adapter diagnostics remain in ignored
`runs/`. The public copy is limited to hashes, booleans, bounds and status;
it contains no endpoint, credential, FEN or teacher label. If the local
originals are removed, the public record remains as a summary but cannot
independently prove the process access boundary.
