# 0004 — Train on pooled top-5 leagues, display one

**Status:** accepted
**Date:** week 1

## Context

The Premier League plays 380 matches a season. Ten seasons is ~3,800 matches, of which a held-out validation set is maybe 400. Later phases add a gradient-boosted ensemble and an agent that proposes features.

## Decision

Train on the pooled top-5 European leagues (E0, SP1, I1, D1, F1) with league as a model feature. Display a single league.

## Reasoning

Pooled, the top five leagues produce ~1,752 matches a season — 380 each for the Premier League, La Liga and Serie A (20 teams), 306 each for the Bundesliga and Ligue 1 (18 teams). Over a decade that's ~17,500 matches versus 3,800.

At 400 validation matches, the difference between a real 1% log-loss improvement and noise is not reliably detectable. This becomes acute once an agent is proposing features in a loop: given enough proposals against a small validation set, something will appear to work purely by chance. More data is the only structural fix; everything else is mitigation.

League-specific effects (home advantage and scoring rates differ measurably across the five) are handled by including league as a feature rather than by fitting separate models, which would put us back at single-league sample sizes.

## Alternatives considered

**Single-league training.** Simpler, avoids cross-league transfer assumptions, and is what most tutorial implementations do. Rejected on sample size.

**Separate model per league.** Cleanest handling of league differences, but reintroduces the sample-size problem it was meant to solve. Rejected.

## Consequences

Ingestion handles five divisions from the start, which makes team-name normalization a week-1 concern rather than a later one. The assumption that team strength is comparable across leagues is a real modelling assumption and should be tested, not asserted — the pooled model needs to be benchmarked against a single-league one rather than assumed better.
