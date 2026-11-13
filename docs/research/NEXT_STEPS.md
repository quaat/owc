# Independent-review handoff and next gate

Start with the [field trial report](M3_FIELD_TRIAL_REPORT.md), then inspect
the [protocol](EXPERIMENT_PROTOCOL.md), [chronology](RESEARCH_LOG.md),
[results](../../reports/experiments/2026-11-13-smoke-pilot/results.json),
and [artifact index](../../reports/experiments/2026-11-13-smoke-pilot/artifact-index.json).

| Reviewer question | Evidence-based answer |
| --- | --- |
| Did Astra genuinely and successfully edit neural weights? | **No.** A genuine nonce inference reached `gpt-6-astra`, but the E0 patch call failed at DNS before a provider response. The actor string in the request is only a declaration. |
| Best observed outcome and dataset? | Among completed static evaluators, the handcrafted heuristic had development MAE 0.2583377147571239 on 28 synthetic positions. No Astra evaluator exists. |
| Did Astra improve over the zero network? | Unmeasured. Zero-network dev MAE was 0.722732457937384; the failed E0 left its hash unchanged. |
| Was any Astra improvement position-specific? | Unmeasured. No accepted patch or behavioral probe after a patch exists. |
| How did Astra compare with baselines? | No direct comparison is possible. Adam finished at MAE 0.704238772392273, handcrafted at 0.2583377147571239, on the same synthetic dev split. |
| Calls, tokens, labels, seconds? | Connection nonce: one successful call, 18 input/11 output tokens, 1.4584902620408684 s. E0: one attempted call, zero completed responses, token usage unavailable, zero teacher disclosures, 1.106812443002127 run seconds. Upstream Stockfish: 308 train and 28 dev labels, 336,381 actual nodes. Adam: 100 distinct training labels, 500 presentations, 20 updates, 0.49710616702213883 training seconds. |
| Repeatable across seeds? | No. Only seed 11 was used for the Adam run and failed E0 attempt. |
| Truly independent held-out evaluation? | No. The generated test split shared the operator's OS permissions and was not scored. |
| Three largest threats to validity? | No successful Astra patch; synthetic, tiny, endgame-free dev data; no OS-isolated held-out service. |
| Most interesting unexpected observation? | The normal threaded `python-chess` UCI wrapper stalled in this sandbox although direct UCI labeling completed, showing how execution isolation can affect a research harness independently of model quality. |
| What should the next experiment test? | One genuine structured Astra patch from a zero model, with full provider transcript, connected-path and material-direction checks, under a process unable to access sealed test data. |
| What must be fixed before a publishable claim? | Licensed representative corpus, genuine patch and replay, three or more seeds, matched resource disclosure, independent frozen-checkpoint testing and uncertainty reporting. |

## Priority order

1. **Immediate engineering correction:** run the adapter in a network-permitted
   process with explicit deployment and exact prompt/source hashes, while an
   independent OS principal protects the test set. Preserve raw provider
   response, usage and M2 checkpoint. Confirm a nonconstant position response.
2. **Controlled experiment:** after that gate, conduct E0 then E5 on a
   representative licensed corpus. Count attempts, accepted/rejected patches,
   label disclosures, model tokens, node budgets, wall time and charges
   separately. Review costs before E20/E50.
3. **Benchmark improvement:** include opening, middlegame, endgame,
   material-imbalance and tactical sources; group split and document license;
   freeze checkpoint selection before blind evaluation.
4. **Replication:** run at least three independent seeds per condition,
   publish trajectories and uncertainty, and make raw artifacts durably
   available under suitable access rules.
5. **Article preparation:** regenerate figures only after direct-edit and
   blind-test data exist; update the [claim register](ARTICLE_EVIDENCE.md)
   before drafting a comparative conclusion.

The single highest-value next action is the first **network-permitted,
test-isolated structured patch call**. It resolves the central missing fact:
whether the configured Astra deployment can produce a valid, position-sensitive
OWC weight revision at all. The current archive is an engineering handoff,
not a completed development experiment or an independently benchmarked study.
