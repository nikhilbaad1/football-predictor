# 0010 — Forward fixtures from the same source, and predictions locked at kick-off

**Status:** accepted
**Date:** week 4

## Context

Until now the project could only re-score history. The landing page invented pairings between highly rated sides because football-data.co.uk's per-season files contain results, not schedules, and `predict_upcoming` had nothing to predict.

That left the central claim in `docs/PLAN.md` section 1 — that this project, unlike most agent portfolios, produces output which "can be graded by the season" — untrue in practice. A backtest grades a model against data that already existed when the model was written. Only a prediction recorded before a match is played demonstrates anything more.

## Decision

Take fixtures from **football-data.co.uk's `fixtures.csv`**, the same publisher as the results, and **store predictions only for matches that have not kicked off**.

## Reasoning

**The same source, because the alternative buys coverage at the cost of a second name vocabulary.** OpenFootball (`docs/PLAN.md` section 3, Tier 1) publishes full-season schedules as JSON, against roughly one week of lookahead from `fixtures.csv`. Full-season coverage is genuinely better. It also introduces a second spelling of every club in Europe, immediately, while entity resolution is still a hand-kept alias table and the agent meant to own that problem does not exist until weeks 4–6.

That trade-off was not hypothetical. Even staying within one publisher, the fixtures feed and the results files disagree: the results say `Ath Madrid`, the fixtures say `Atl. Madrid`. Unaliased, that split one club into two rows, and the first live prediction it produced gave Atletico Madrid a 49.7% chance at home against Malaga where the market's opening price implied about 73%. Nothing failed. The output was well-formed, three outcomes summing to one, and meaningless — the model was rating a club it had never seen. If a single extra spelling inside one source does that, adopting a second source's entire vocabulary a week before the resolution agent lands is the wrong order to do things in.

`fixtures.csv` also carries the division codes already in use, the same `dd/mm/yyyy` convention, and needs no new fetching or caching policy. OpenFootball remains the obvious upgrade once entity resolution is an agent's job rather than a dictionary.

**A prediction is evidence only if it could not have been rewritten.** `store_predictions` drops rows whose `match_date` has passed. Re-running the predictor is normal and should refresh a fixture's numbers as new results arrive, but only while the match is still ahead. Once it has kicked off, the stored row is the record of what was forecast without knowing the answer. Allowing a re-run to overwrite it would silently convert the forward record into a backtest with extra steps — and it would do so invisibly, since the row would still look like a prediction with a timestamp.

Locking on date rather than adding a schema column keeps this to one filter with a stated reason. Kick-off time exists in the feed and is not stored; the practical cost is that a prediction can still be refreshed on match day before kick-off, which is acceptable and is noted as a limitation.

**`ON CONFLICT DO NOTHING`, never `DO UPDATE`.** Fixtures share a uniqueness key with results. The results ingester upserts goals and result, and copying that pattern here would write NULLs over finished matches whenever a fixture row collided with a played one — corrupting the backtest silently. `tests/test_fixtures.py::TestNeverOverwritesAResult` asserts this, and was checked by mutation: changing the clause to `DO UPDATE` makes it fail.

**No odds are taken from the fixtures feed.** It carries opening prices only; the C-suffixed closing columns are present in the header but empty, because a closing line does not exist until shortly before kick-off. Storing openings would put a weaker number exactly where the benchmark expects the informed one, which is the trap `CLAUDE.md` already warns about. Odds arrive later, with the result, from the season file.

## Alternatives considered

**OpenFootball for full-season schedules.** Better coverage, rejected for now on the entity-resolution grounds above. Revisit in weeks 4–6 alongside the ingestion agent, which is the right owner for a second vocabulary.

**Refuse to predict a team with no match history.** Tempting after the Atletico incident, and rejected: a genuinely promoted club has no history either, and the code cannot tell the two apart. Raising would break ingestion every August for the honest reason while catching alias bugs only incidentally. Instead `teams_without_history` surfaces them and `scripts/fixtures.py` prints a warning naming the teams, so an alias gap is loud rather than silent.

**Let the page keep showing invented pairings when nothing is scheduled.** Rejected as the only genuinely dishonest thing the page could do. The feed is legitimately empty between rounds and before a season opens, so the fallback stays — but it renders behind a notice saying the matches are illustrative and not real upcoming fixtures.

## Consequences

The Premier League shows nothing until the source begins publishing its fixtures for the season; on the day this landed, `fixtures.csv` carried two La Liga matches and one League One match and no `E0` at all, and the `2627` `E0` results file did not yet exist. The pipeline was verified end to end against La Liga instead. This is a property of the source's publishing schedule, not a defect, and it is why the illustrative fallback and its notice exist.

`DEFAULT_SEASONS` is now generated from `FIRST_SEASON_START` to the current season rather than hand-listed. A hardcoded list silently stops covering the current season once the calendar rolls over, and a fixture ingested for a season whose results file is never fetched would stay unplayed forever.

The display page recomputes probabilities from the fit held in memory rather than reading the stored rows. The stored table is the durable, locked record used for grading; the page is a view. Both run the same `MODEL_VERSION` and agree unless the data moved between runs. Serving the page directly from the locked rows is the cleaner end state and waits on a scheduled refit.

Postponements leave unplayed rows in the past. These are reported by `stale_fixtures` and removed only by an explicit `--prune`, because deleting a match also deletes what was predicted for it, and that record is the evidence this ADR exists to protect.
