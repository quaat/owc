# OWC article workspace

This directory contains the living, independently reviewed article draft for
[Open Weight Chess](https://github.com/quaat/owc).

- [Proposed comparative study](STUDY_DESIGN.md): prospective gates, sampling,
  fairness, blinding, budgets and publication criteria (**not preregistered**).
- [Preliminary manuscript](OWC_MANUSCRIPT.md): experimental protocol, first
  integration attempt, measured smoke results, limitations and verified
  references.
- [Claim/evidence register](../research/ARTICLE_EVIDENCE.md): establishes which
  claims are currently supportable and which require new results.
- [Primary M3 field trial report](../research/M3_FIELD_TRIAL_REPORT.md) and
  [machine-readable results](../../reports/experiments/2026-11-13-smoke-pilot/results.json):
  the source for all quoted OWC experiment metrics.

**Current scientific status:** engineering smoke and failed E0 integration.
No accepted Astra-generated patch, no blind-test result, no multi-seed direct
experiment, and no playing-strength claim. This draft is not publication ready.

**Process:** Work on the experiment/adapter PR and the article PR independently.
When a new real run is approved and archived, update the immutable evidence
register and technical report before amending the manuscript. Treat missing
costs, missing labels and unavailable results as missing, not zero. Do not put
secrets, private Foundry URLs, restricted corpora or raw sensitive transcripts
in this public directory.
