# 0008 — Freeze the deterministic baseline; don't tune the gap away

**Status:** accepted
**Date:** week 3

## Context

The phase-1 slice is complete and has been backtested on real football-data.co.uk data for the first time — every number before this came from synthetic data generated to exercise the pipeline. Premier League, 2017-09 to 2026-05, 3,381 matches scored, models fitted on all five leagues pooled.

```
              model    n  log_loss  brier    ece  accuracy
                elo 3381    0.9774 0.5802 0.0171    0.5350
        dixon_coles 3381    0.9765 0.5777 0.0095    0.5413
              blend 3381    0.9710 0.5763 0.0113    0.5368
 baseline_base_rate 3381    1.0661 0.6451 0.0000    0.4413
 baseline_bookmaker 2660    0.9639 0.5717 0.0047    0.5496
blend (odds subset) 2660    0.9826 0.5844 0.0114    0.5293

gap to bookmaker: +0.0188 log-loss
```

The synthetic-data README predicted "roughly 0.96–0.99 for a top league, against a bookmaker around 0.95–0.97." The real figures are 0.971 and 0.964. The harness predicted its own real-world result before seeing real data, which is the strongest available evidence that it is not deceiving us.

## Decision

Treat these numbers as the frozen phase-1 baseline. Stop work on the deterministic models: no fitted blend weight, no isotonic calibration layer, no hyperparameter search over the Elo constant or the Dixon-Coles half-life. Move to phase 2.

Any future model change is measured against this table, and the table is not re-run to make a change look good.

## Reasoning

**The gap to the bookmaker is stable across seasons and does not respond to training volume.** Per season, from 2019/20 onward:

| season | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| gap | 0.005 | 0.024 | 0.018 | 0.026 | 0.027 | 0.013 | 0.016 |

No trend, across a stretch where the pooled training set roughly triples. If the deficit were a sample-size problem it would shrink; it doesn't. So the remaining distance is not something more data or a better-tuned constant recovers — it is information the models structurally cannot see: lineups, injuries, xG, rest days, fixture congestion. Those arrive in later phases. Spending week 3 hunting for 0.002 of log-loss inside a model that cannot see a striker is the wrong place to work.

**Tuning here would violate the project's own non-negotiable.** `CLAUDE.md` states the bookmaker is the benchmark, not the target, and that tuning until we "win" means we have overfit. With 3,381 scored matches, and having now looked at the results, any further adjustment chosen by watching this table is selection on the test set. The honest move is to declare the number and stop.

**The blend is doing more work than the log-loss improvement suggests, and that argues against replacing it with a fitted weight.** Dixon-Coles put an outcome under 1% in 13 of 3,381 matches — and that outcome occurred 3 times, which is badly calibrated in the tail (a genuine 1% event should land about 0.1 times in 13). The 35/65 blend absorbs it: the blend's smallest probability anywhere in the window is 0.0097, and flooring blend probabilities anywhere from 0.5% to 3% moves log-loss by less than 0.0001. The worst three matches account for 0.3% of total loss. So the fixed blend weight is functioning as variance reduction against degenerate Dixon-Coles fits, not merely as an averaging gain — and a weight fitted to minimise log-loss on this window would be optimising for the average case while discarding that protection.

## Alternatives considered

**Fit the blend weight on a validation split.** The obvious next tweak. Rejected: the improvement available is smaller than the noise in a validation set this size, and as above, the fixed weight buys tail protection that a log-loss-optimal weight would give up. Revisit only when a third model (the gradient-boosted layer) makes the blend weight a genuine three-way decision rather than a scalar.

**Add an isotonic calibration layer.** Tempting, because two calibration bins look poor: 0.4-0.5 is overconfident by 0.036 and 0.9-1.0 by 0.052. But 0.9-1.0 holds 22 observations, where one match moves the gap by 0.05, and the three bins carrying two-thirds of the mass are already within 0.01. Isotonic regression fitted on this data would mostly be fitting those thin bins. Deferred until there is a model whose miscalibration is visible in the *large* bins.

**Raise `--min-train` above 2,000.** 2,000 pooled matches is barely over one pooled season, and it shows: in the first scored season Dixon-Coles (0.9770) is worse than Elo (0.9697), consistent with a thin fit, and reverses the following season. Rejected as a change to make now because it would shorten the scored window and therefore change the baseline it is being measured against. Noted as a known property instead.

## Consequences

The README results table is now real and dated, and carries the odds-coverage caveat: closing odds exist only from 2019/20, so the bookmaker benchmark rests on 2,660 of the 3,381 matches, not all of them.

Phase 2 work — fixtures feed, MCP server, ingestion and entity-resolution agents — is unblocked and does not need to wait on further model work.

Because the underlying CSVs grow as seasons complete, re-running the backtest later will not reproduce this table exactly. The numbers here are a snapshot as of week 3, not an invariant. When they are regenerated, the regenerated figures go in the README and this ADR stays as written.
