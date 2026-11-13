# Issue #19: bounded feedback and Astra adapter review

This change addresses the PR #18 request-history review finding. It does not
contain a new provider call or alter the November 13 smoke artifacts.

## Request and audit contract

The runner and adapter use the same canonical UTF-8 JSON serializer and a
256,000-byte input limit. The runner reserves 4,096 bytes of headroom. It
sends at most five compact prior-round summaries, omitting oldest summaries
until the **entire** request fits. Each summary retains exact before/after
model hashes, a SHA-256 of the full patch, its reason, operation kinds and
counts, edited/changed cell counts, and feedback that was already disclosed.
Full patches, requests and outcomes stay in the ignored local `events.jsonl`
and immutable weight revisions. An oversized request that cannot be reduced
raises a recorded `RequestSizeError` before invoking the adapter. Feedback
counts increase only when a bounded request is about to be sent.

The event log records actual request bytes and SHA-256 for each accepted
round. E0 has no teacher targets; later requests may carry only charged
development labels. `dev_mae_undisclosed` is visible to the orchestrator in
its local event log, not in the request sent to the editor. It is a
development metric, not a blind held-out result. `CommandAgent` is still an
operator-chosen external command and requires OS isolation from held-out data.

## Adapter provenance and diagnostics

Before a provider request, the adapter requires a diagnostic directory owned
by its OS user with no group or world access. It hashes the exact agent
request, provider wire request, instructions and adapter source, and persists
a 0600 preflight record of the request bytes and hashes before network I/O. It stores
exact provider request/response bytes, IDs, usage, the requested deployment,
reported model, and failure detail in a 0600 restricted transcript. A paired
0600 `.public.json` file contains hashes, bounded identifiers, token counts,
elapsed time and a coarse failure code. It contains no endpoint, raw request,
raw response or positions. Review that attestation before copying it into a
public archive. A diagnostic write failure exits without emitting a patch;
a partial restricted file may remain for operator recovery.

The adapter accepts an exact reported-model match or one explicit
`OWC_ASTRA_ALLOWED_RESPONSE_MODEL` mapping set by the operator. It records
both names and the policy. An HTTPS response from the configured Foundry
endpoint and an operator mapping do not independently prove the underlying
model identity. The requested deployment must still be verified during the
first real E0 trial. The adapter has a socket timeout; the runner also has
an overall subprocess wall-clock limit. The attestation remains local until
the #20 run manifest is assembled and checked.

Offline tests cover a legal near-limit rank-one patch across six rounds,
wire-size accounting, an oversized request rejected before invocation,
restricted file permissions, failed diagnostic writes, model-name mismatch,
public identifier redaction and provider timeout. The fixture has no chess
intelligence. No E5 or blind evaluation was performed for this fix.
