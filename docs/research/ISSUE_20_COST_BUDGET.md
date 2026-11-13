# Issue #20 — published pricing and one-call E0 cost gate

**Decision status:** Conditional operational envelope, **not** proof of the deployment's billing SKU or a guarantee against all Azure charges. No provider inference call was performed by preparing this document.

## Official pricing source

Microsoft's [GPT-6 pricing and deployment options](https://azure.microsoft.com/en-us/blog/gpt-6-astra-sol-and-luna-for-production-agents-in-microsoft-foundry/) lists, among other Standard tariffs:

| GPT-6 Astra Standard deployment | Input / 1M tokens | Output / 1M tokens |
| --- | ---: | ---: |
| Global, short context | $10.00 | $50.00 |
| Global, long context | $20.00 | $75.00 |
| Data Zone EU, short context | $12.00 | $60.00 |
| Data Zone EU, long context | $24.00 | $90.00 |

These are **published public rates**, not a verified live quote for the current `oceanai-dev-swe-01-fo` deployment. Provisioned/Priority and other billing arrangements may differ. Check the real Azure deployment SKU and pricing meter before the paid call.

## Cost envelope for the first E0 only

The isolated launcher supplies:

- at most **one model-inference call**, with no automatic retries;
- model `gpt-6-astra`, zero teacher-label disclosures;
- **32,000 UTF-8 bytes** maximum for the complete serialized **provider JSON request**, enforced before network I/O (the ordinary agent request protocol cap remains 256,000 bytes);
- **3,000 output tokens** maximum;
- 90-second provider socket timeout and 150-second experiment wall-clock budget.

Conservatively using **32,000 input tokens** plus 3,000 output tokens at the published **EU Data Zone Standard long-context** prices gives:

```text
32,000 * $24 / 1,000,000 + 3,000 * $90 / 1,000,000 = $1.038 USD
```

This is an **illustrative conservative tariff estimate** for standard token charges. Actual billable token counts, hidden prompt overhead, caching treatment, selected tier, minimum charges, taxes, and post-processing can differ. Byte count is not a provider-supplied token count, and the estimated cost is **not** a live hard spend limit. Microsoft Cost Management may lag real usage.

**Proposed operator project ceiling: $2 USD estimated total Standard input/output token charges for the single E0 call.** This is a conditional guard, not a claim that a price or billing meter has already been verified. If the actual deployment is not an applicable Standard pay-as-you-go tier, the calculation does not apply and the run must stop pending a real price check. Do not run E5/E20/E50 as part of this decision.

## Required call sequence

1. Confirm the actual deployment SKU/region/rate from Foundry/Azure deployment metadata or approved billing contract, without exposing credentials; record its evidence and date in a restricted note.
2. Verify the **full provider request bytes** meet 32,000 and the predicted charge is under the approved ceiling. If the intended one-call budget is higher than $2 or cannot be bounded, **stop and request a new budget decision**.
3. Re-run the restricted-namespace access-denial and authenticated model-list probe from the actual runtime. Confirm no unapproved held-out data can be read by the model process.
4. Freeze source/prompt/patch hashes; make no more than **one** E0 inference request, no retries; retain 0600 raw preflight and provider transcript, return the sanitized attestation, and measure actual reported tokens/time.
5. Record a genuine accepted patch and position-sensitive behavior, or record the exact failure with no invented result. **Do not begin E5 until independent review.**

The [first synthetic E0 preparation record](../../reports/experiments/2026-11-13-e0-preparation/preparation.json) was created **before** this published-price analysis. Its `dollar_cost_ceiling: null` remains a historically correct original datum; do not rewrite it retroactively.
