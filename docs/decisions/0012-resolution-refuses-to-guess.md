# 0012 — Entity resolution refuses to guess, and availability is a time series

**Status:** accepted
**Date:** week 4

## Context

The Fantasy Premier League API is the project's second data source and the first with player-level data: 20 teams, 592 players, and — the reason ADR 0002 put the Premier League first — `status`, `chance_of_playing_next_round` and a `news` string per player. That is the expected-minutes input PLAN §5.3 needs.

It also arrives in a third spelling of every club. The results files say `Tottenham`, the fixtures feed says `Atl. Madrid`, FPL says `Spurs` and `Man Utd`. ADR 0010 already recorded what happens when that goes unnoticed: `Ath Madrid` against `Atl. Madrid` split one club in two and produced a 49.7% home probability for a side the market priced near 73%. Well-formed, three outcomes summing to one, and meaningless.

Measured against the existing alias table, 17 of FPL's 20 team names already resolved. The three that did not are three different problems:

| FPL name | Reality | What a nearest-match rule does |
|---|---|---|
| `Hull City` | This database has `Hull` — one club | Correct |
| `Ipswich Town` | This database has `Ipswich` — one club | Correct |
| `Coventry City` | No history here at all | **Matches it to `Leicester City`** |

## Decision

**Resolve only what can be justified, and refuse the rest.** The resolver tries exact identity, then the alias table, then a head-word match that ignores club-type suffixes. Anything else is written to `name_resolutions` with its near misses attached and left undecided, and the club's players are not ingested.

**Availability is stored as a dated time series**, not as columns on `players`.

## Reasoning

**No string metric separates the three cases above, because the difference is not in the strings.** `Hull City`/`Hull` and `Coventry City`/`Leicester City` are comparably similar as text. What distinguishes them is knowing which football clubs exist — world knowledge, not edit distance. So any similarity threshold loose enough to fix Hull and Ipswich also merges Coventry City into Leicester City, and that failure is far worse than the one it fixes: a missing club is visible as a gap, while a wrongly merged one produces rows that all look valid and a model confidently rating a team that does not exist.

The head-word rule is the most that can be justified without world knowledge. Stripping club-type words (`City`, `Town`, `United`, `FC`…) leaves the part that identifies the club: `Hull City` → `hull`, `Ipswich Town` → `ipswich`. That resolves both real cases. It also maps `Coventry City` → `coventry` and `Leicester City` → `leicester`, which do not match — the suffix is exactly what those two share, and the head is what tells them apart. The rule additionally refuses when a head word maps to more than one known club, since choosing between them would be arbitrary.

Result: **19 of 20 resolved, and the single refusal is the one club that is genuinely new.** No false matches.

**The resolver can never output "this is a new club."** It can only report that nothing matched. Distinguishing "new club" from "unrecognised spelling of an existing one" is precisely the judgement it lacks, so `Resolution.decision` is `"matched"` or nothing. That absence is the entity-resolution agent's job, and this ADR fixes the baseline it has to beat: 19/20, zero wrong.

**Candidates are surfaced loosely and are never decisions.** `Coventry City` comes back with `['Leicester City', 'Manchester City', 'Norwich City']` at a deliberately low similarity cutoff. Nothing is auto-accepted from that list, so recall matters and precision does not — an agent concluding "these are three different clubs, therefore this is a new one" needs to see them. A tighter cutoff surfaced nothing at all, which tells the next reader less.

**Availability is a time series because it feeds prediction.** A player injured today was available last week. A single mutable `is_injured` column would mean any historical fit sees today's knowledge attached to last week's match — leakage, and the exact failure non-negotiable #1 exists to prevent. Each run writes a snapshot stamped with its date; re-running the same day refreshes that day's row and leaves earlier ones untouched. `TestAvailabilityIsATimeSeries` asserts that yesterday's snapshot still reads "available" after today's says "injured".

**An unresolved club costs its squad.** Coventry City's 32 players are skipped rather than attached to a guess. Missing players are recoverable once the name is decided; misattributed ones are not, because nothing downstream ever surfaces them.

## Alternatives considered

**A similarity threshold tuned on the three known cases.** The obvious approach, and it is unfixable rather than merely imperfect: the cases are not separable in the feature it uses. Tuning the threshold trades Hull against Coventry in both directions and no value gets all three right.

**Resolve to the nearest match and flag low-confidence rows for review.** Rejected because flagged-for-review rows do not get reviewed once the pipeline is running, and by then the merge is already in the data being trained on. Refusing up front costs a visible gap instead of an invisible corruption.

**Create a new team row whenever nothing matches.** This would have handled Coventry City correctly and silently created a duplicate for any unrecognised spelling — the `Atl. Madrid` bug, automated.

**Store availability as columns on `players`.** Simpler and leaks. See above.

## Consequences

`name_resolutions` accumulates the decisions this layer *did* make (10 exact, 7 alias, 2 head-word) alongside what it declined. Those known-good rows are the eval set the agent gets scored against, not just an audit log.

`players` and `player_availability` exist but nothing consumes them yet. Expected minutes enters the model in weeks 7–12; this phase establishes the source and its history so there is something to fit on when it does.

The FPL API is Premier League only. The other four leagues have players in no source the project uses, so player-level features will be single-league for the foreseeable future while match-level ones stay pooled (ADR 0004).

`fpl_id` is unique per season but not stable across them. Nothing yet depends on cross-season player identity; when it does, that is a second entity-resolution problem and it will need its own answer.
