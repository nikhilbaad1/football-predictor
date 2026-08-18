# Football Predictor

Match outcome prediction for Europe's top five leagues, using Elo and Dixon-Coles fitted on free historical data, evaluated honestly against the bookmaker's closing line.

**Status: week 4 — deterministic models, a forward-fixture feed, an MCP tool layer, FPL player data, and CI plus guardrails. No agents yet.** This is the foundation for a multi-agent system; the roadmap is at the bottom.

> Predictions are probabilistic estimates from historical results. Not guarantees, and not betting advice.

## What it does

Given a fixture, it returns three probabilities that sum to 1 — home win, draw, away win — plus expected goals and a full scoreline distribution.

```
$ python scripts/predict.py "Arsenal" "Chelsea"

Arsenal v Chelsea
  home win   67.1%
  draw       20.9%
  away win   11.9%

  expected goals  1.98 - 0.74
  likeliest score 2-0 (12.9%)

  elo  1913 v 1636

  Probabilistic estimate, not betting advice.
```

There is no "win probability" scalar anywhere. Roughly a quarter of top-league matches are draws, and a model that collapses to two outcomes misprices essentially everything.

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env

python scripts/ingest.py        # downloads ~10 seasons x 5 leagues, caches to data/raw/
python scripts/backtest.py      # walk-forward evaluation vs the bookmaker
python scripts/fixtures.py      # upcoming fixtures, predicted and recorded
python scripts/fpl.py           # players + injury/availability snapshot
uvicorn fpp.api:app --reload    # http://localhost:8000
```

Ingest takes a few minutes on first run and is instant afterwards — files are cached and never re-fetched. `fixtures.py` is the exception and deliberately never caches: matches get postponed and rescheduled, so a stale fixture list is worse than none. Run it on a schedule.

## Predicting forward

A backtest grades a model against data that already existed when the model was written. `scripts/fixtures.py` pulls the coming week's fixtures, predicts them, and writes those probabilities to the database **before the matches are played**:

```
$ python scripts/fixtures.py --division SP1

new fixtures: SP1 2

La Liga — 2 fixture(s), 2 stored

  2026-08-19  Atletico Madrid          v Malaga       64.1% / 21.7% / 14.2%
  2026-08-20  Rayo Vallecano           v Alaves       47.4% / 28.2% / 24.3%
```

Stored predictions are locked once a match's date has passed. Re-running refreshes a fixture while it is still ahead, and refuses to touch it afterwards — a prediction that can be rewritten after the result is known is not evidence of anything. See [0010](docs/decisions/0010-forward-fixtures-and-locked-predictions.md).

## Resolving clubs across sources

Every source spells clubs differently. The results files say `Tottenham`, their own fixtures feed says `Atl. Madrid` where the results say `Ath Madrid`, and the FPL API says `Spurs`. Getting this wrong doesn't raise — it splits one club into two rows, and the model then rates a team it has never seen. That happened once already, and produced a 49.7% home probability for a side the market priced near 73%.

Measured against FPL's 20 team names, the resolver settles 19 and **refuses the twentieth**:

```
$ python scripts/fpl.py --pending

1 name(s) awaiting a decision:

  [fpl/team] 'Coventry City'
      No exact, alias, or head-word match. String similarity alone cannot say
      whether this is a new club or a new spelling of an existing one.
```

`Hull City` → `Hull` and `Ipswich Town` → `Ipswich` resolve, because club-type suffixes say what kind of club it is rather than which one. `Coventry City` doesn't, and that is the point: any threshold loose enough to fix the first two also matches Coventry City to **Leicester City**. The two cases are not separable as text — what tells them apart is knowing which clubs exist. So the resolver reports what it cannot justify instead of inventing a link, and the unresolved club's players are skipped rather than attached to a guess. See [0012](docs/decisions/0012-resolution-refuses-to-guess.md).

That queue is the entity-resolution agent's first job, with a baseline it has to beat: 19 of 20, zero wrong.

## Asking the database questions

An MCP server exposes the data to Claude as tools. `.mcp.json` is checked in, so Claude Code connects automatically — on macOS or Linux change the interpreter path to `.venv/bin/python`.

| Tool | Answers |
|---|---|
| `predict_match` | Three-way probabilities for any pairing, real or hypothetical |
| `get_upcoming_fixtures` | Scheduled fixtures and the predictions locked in before kick-off |
| `get_head_to_head` | Past meetings, with de-vigged closing prices where they exist |
| `get_team_form` | Recent results and current Elo for one club |
| `run_sql` | Read-only escape hatch for what the four don't cover |

Four purposeful tools rather than one `execute_sql`, because a raw-SQL tool pushes schema knowledge onto the caller, makes every call a possible table scan, and leaves nothing to evaluate. Team names resolve through the alias table, and an unknown one comes back as `No team called 'Arsenl'. Closest matches: Arsenal.` — an error you can act on beats a separate lookup tool.

**The server cannot write.** The connection is opened read-only, so a write fails inside the driver. That matters because the `PreToolUse` guardrail inspects shell commands and cannot see an MCP call — this layer has to hold on its own. See [0011](docs/decisions/0011-mcp-tools-over-raw-sql.md).

```bash
python -m fpp.mcp_server    # stdio, which is how Claude Code connects
```

## How it works

**Elo** tracks team strength, updating after each match with a home-advantage term and a damped margin-of-victory multiplier. Elo alone yields an expected *score*, not three probabilities, so the split into home/draw/away is fitted with a multinomial logistic regression on rating difference rather than assumed. That lets the model learn that draws get less likely as the gap widens, instead of hard-coding a draw rate.

**Dixon-Coles** ([1997](https://academic.oup.com/jrsssc/article/46/2/265/6990310)) fits attack and defence strengths per team and models goals as Poisson:

```
lambda = exp(attack_home - defence_away + home_advantage)
mu     = exp(attack_away - defence_home)
```

Independent Poisson gets low scores wrong — it under-predicts 0-0 and 1-0 and over-predicts 1-1. Dixon-Coles corrects the four scorelines where both teams score at most once, via a parameter `rho` that fits around -0.1 for football. Matches are exponentially down-weighted by age (default half-life 180 days) so recent form dominates.

The fit supplies an **analytic gradient**. With ~150 teams the parameter vector is ~300 long, so a numerical gradient would need ~300 likelihood evaluations per optimizer step — too slow to refit repeatedly inside a backtest. `tests/test_dixon_coles.py` checks it against `scipy.optimize.approx_fprime`.

The two models are blended 35/65 in favour of Dixon-Coles. That split is a prior, not a fitted value — with a validation set this size, fitting blend weights costs more than the weights are worth.

## Evaluation

The one invariant that matters: **a prediction for a match on date D uses only matches strictly before D.** This is enforced structurally in `evaluation/backtest.py` and verified by a test that corrupts every result *after* a given block and asserts the predictions for that block are bit-for-bit unchanged. If that test fails, every number the project reports is wrong.

Metrics are proper scoring rules — log-loss and multiclass Brier — plus expected calibration error. Accuracy is reported for context but is not a headline metric: a model that always picks the favourite scores well on accuracy and is useless.

Three baselines, all of which must be beaten or explained:

| Baseline | What it is |
|---|---|
| Uniform | 1/3 each. Log-loss = log 3 ≈ 1.0986. |
| Base rate | Constant at observed frequencies. Deliberately handed the answer key — failing to beat it means nothing was learned. |
| **Bookmaker** | De-vigged closing odds. The real benchmark. |

### Results

Premier League matches from 2017-09 to 2026-05, walk-forward, models fitted on all five leagues pooled. Initial train 2,000 matches, refit every 40.

```
              model    n  log_loss  brier    ece  accuracy
                elo 3381    0.9774 0.5802 0.0171    0.5350
        dixon_coles 3381    0.9765 0.5777 0.0095    0.5413
              blend 3381    0.9710 0.5763 0.0113    0.5368
 baseline_base_rate 3381    1.0661 0.6451 0.0000    0.4413
 baseline_bookmaker 2660    0.9639 0.5717 0.0047    0.5496
blend (odds subset) 2660    0.9826 0.5844 0.0114    0.5293

gap to bookmaker: +0.0188 log-loss
  Behind the closing line, as expected.
```

Reproduce with `python scripts/backtest.py`. The blend beats base rates by 0.095 log-loss, sits 0.019 behind the de-vigged closing line, and is calibrated to within 0.011 ECE. Dixon-Coles calibrates better than Elo (0.010 vs 0.017), the expected consequence of modelling scorelines directly instead of fitting a mapping from rating difference.

Accuracy, on the 2,660 matches both are scored on, is 52.9% for the blend against the bookmaker's 55.0%. Both are far below the 63% the synthetic development data suggested — 53% is the right order of magnitude for real top-flight football, and a reminder of why accuracy is not the headline metric.

**Calibration** (blend, pooled across the three outcomes):

```
    bin    n  predicted  observed    gap
0.0-0.1  543      0.064     0.087 -0.022
0.1-0.2 1596      0.156     0.165 -0.009
0.2-0.3 3731      0.254     0.256 -0.003
0.3-0.4 1343      0.345     0.357 -0.012
0.4-0.5 1040      0.448     0.412  0.036
0.5-0.6  772      0.548     0.539  0.009
0.6-0.7  572      0.649     0.654 -0.005
0.7-0.8  355      0.747     0.713  0.034
0.8-0.9  169      0.839     0.852 -0.013
0.9-1.0   22      0.916     0.864  0.052
```

The three bins holding two-thirds of the mass agree with observed frequency to within 0.01. The visibly worse bins are the thin ones — 0.9-1.0 has 22 observations, where a gap of 0.05 is one match.

### Reading the results table

**Why `blend (odds subset)` scores worse than `blend`.** This source carries closing odds only from 2019/20 onward, so the 721 matches without them are seasons 2017/18 and 2018/19 — and 2018/19 was the most predictable season in the window (blend log-loss 0.898 against a 0.971 average). The subset is a harder sample, not a different model. The gap is computed on the subset precisely because it is the only comparison where both sides are scored on the same matches.

**The gap does not close as training data accumulates.** Per season it runs 0.005, 0.024, 0.018, 0.026, 0.027, 0.013, 0.016 — no trend, over a stretch where the pooled training set roughly triples. So the distance to the closing line is not a sample-size problem, and it will not be tuned away. It is information these models cannot see: lineups, injuries, xG, rest days. That is the case for the later phases, now measured rather than assumed.

**We do not expect to beat the bookmaker.** Closing odds aggregate an enormous amount of information and are the hardest benchmark in the field. If the gap ever goes negative, the first hypothesis is a leakage bug, not an edge.

## Data

[football-data.co.uk](https://www.football-data.co.uk/data.php) — free CSVs, no key, no scraping, results plus closing odds from 15+ bookmakers back to 1993. Player identity and availability come from the [Fantasy Premier League API](https://fantasy.premierleague.com/api/bootstrap-static/) — official, free, no key, and the only free source carrying injury and expected-availability data ([0002](docs/decisions/0002-premier-league-first.md)).

Results and fixtures come from the same publisher, which keeps team names and date conventions consistent. Traps in this source, all handled and all tested:

- Dates are `dd/mm/yyyy`. Parsed without `dayfirst=True`, `05/08/2024` becomes 8 May and the whole dataset silently reorders.
- `B365H` is the **opening** price; the closing line is `B365CH`. Only closing odds are stored, so the opening price cannot be used by accident.

Models train on all five leagues pooled and display one. Single-league data is ~380 matches a season; pooled is ~1,750. At single-league volume most apparent improvements are noise, which becomes acute once an agent is proposing features in a loop.

## Layout

```
src/fpp/
  ingest/     football-data.co.uk results + fixtures, FPL players, name resolution
  models/     elo.py, dixon_coles.py, blend.py
  evaluation/ metrics.py, backtest.py
  mcp_server/ MCP tools over the DB (read-only)
  api.py      FastAPI: JSON endpoints + one server-rendered page
scripts/      ingest, backtest, predict, fixtures, fpl
docs/decisions/  ADRs — why things are the way they are
tests/        165 tests; model maths checked against closed-form values
.github/workflows/  CI: lint, tests on 3.10 and 3.13, leakage check as its own job
.claude/hooks/      PreToolUse guard: no pushes or merges to main, no destructive SQL
```

Work reaches `main` through a pull request. That is enforced by the hook rather than
by convention, because the agents arriving in phase 2 are supposed to stop at a PR and
a prompt is not what makes them stop — see [0009](docs/decisions/0009-agent-guardrails-are-structural.md).

## Testing

```bash
pytest tests/ -q     # 165 passed
```

Tests assert against known truth, not stored snapshots. Synthetic data is generated from *known* team strengths, so the tests check that the model recovers them — a failure should always be explainable as "the model is now wrong about X", never "a number moved". The Elo update is checked against the closed-form 400-point/10:1 property; the Dixon-Coles `tau` against the paper's definition; the gradient against a numerical one.

## Decisions

See [`docs/decisions/`](docs/decisions/). Some that shaped this:

- [0002](docs/decisions/0002-premier-league-first.md) — Premier League before La Liga, because the FPL API is the only free source with injury and expected-availability data.
- [0003](docs/decisions/0003-sql-for-structured-rag-for-text.md) — RAG is for unstructured text, not for the match database. Embedding rows to retrieve them by cosine similarity converts an exact lookup into an approximate one.
- [0007](docs/decisions/0007-squad-similarity-is-sql.md) — cosine similarity between two one-hot starting XIs is `shared/11`. That's a join and a count, not a vector index.
- [0008](docs/decisions/0008-freeze-the-deterministic-baseline.md) — the deterministic baseline is frozen at the numbers above. The gap to the closing line is stable across seasons, so closing it by tuning would be overfitting, not progress.

## Known limitations

- **The bookmaker baseline covers 2019/20 onward only** — 2,660 of the 3,381 scored matches. Closing-odds columns do not exist in the earlier CSVs, so the headline benchmark rests on seven seasons, not nine. Any comparison against it inherits that window.
- **Fixture lookahead is about a week.** The feed covers roughly the next seven days, so between rounds and before a season opens there is genuinely nothing scheduled and the page falls back to illustrative pairings, labelled as such. Full-season schedules from OpenFootball wait on the entity-resolution agent, which is the right owner for a second source's team names ([0010](docs/decisions/0010-forward-fixtures-and-locked-predictions.md)).
- **Predictions can still be refreshed on match day.** They are locked once the match *date* has passed, not once the whistle blows — kick-off time is in the feed but not stored.
- **No player-level model yet.** Goal probability, ratings, and head-to-head form are weeks 7–12.
- **De-vigging is multiplicative**, which removes margin evenly across outcomes. Real books load more onto longshots, so this slightly overstates unlikely outcomes. Shin's method would be more accurate if the baseline comparison becomes load-bearing.
- **Blend weights are a prior, not fitted.**
- **Promoted teams start at the league average**, so early-season predictions for them are weak.

## Roadmap

| Phase | Scope |
|---|---|
| **1 (done)** | Ingestion, Elo + Dixon-Coles, evaluation harness, API and page |
| 2 | MCP server over the DB (done); ingestion/entity-resolution agent; maintenance agent on GitHub Actions |
| 3 | RAG over football news and match reports; hybrid retrieval; Ragas evals in CI |
| 4 | Gradient-boosted ensemble and player models, built by an experimentation agent under an evaluator-optimizer loop |
| 5 | Scraper-repair agent, more leagues, deployment |

## Attribution

Results and closing odds from [football-data.co.uk](https://www.football-data.co.uk/). Dixon, M.J. and Coles, S.G. (1997), "Modelling Association Football Scores and Inefficiencies in the Football Betting Market", *JRSS-C* 46(2), 265–280.
