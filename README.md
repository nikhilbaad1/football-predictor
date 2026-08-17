# Football Predictor

Match outcome prediction for Europe's top five leagues, using Elo and Dixon-Coles fitted on free historical data, evaluated honestly against the bookmaker's closing line.

**Status: week 1–3 slice — deterministic models, no agents yet.** This is the foundation for a multi-agent system; the roadmap is at the bottom.

> Predictions are probabilistic estimates from historical results. Not guarantees, and not betting advice.

## What it does

Given a fixture, it returns three probabilities that sum to 1 — home win, draw, away win — plus expected goals and a full scoreline distribution.

```
$ python scripts/predict.py "Arsenal" "Chelsea"

Arsenal v Chelsea
  home win   54.9%
  draw       21.6%
  away win   23.5%

  expected goals  1.94 - 1.27
  likeliest score 1-1 (10.3%)
  elo  1913 v 1810
```

There is no "win probability" scalar anywhere. Roughly a quarter of top-league matches are draws, and a model that collapses to two outcomes misprices essentially everything.

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env

python scripts/ingest.py        # downloads ~10 seasons x 5 leagues, caches to data/raw/
python scripts/backtest.py      # walk-forward evaluation vs the bookmaker
uvicorn fpp.api:app --reload    # http://localhost:8000
```

Ingest takes a few minutes on first run and is instant afterwards — files are cached and never re-fetched.

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

### Sample output

```
              model    n  log_loss  brier    ece  accuracy
                elo 2508    0.8653 0.5012 0.0165    0.6304
        dixon_coles 2508    0.8639 0.5013 0.0052    0.6300
              blend 2508    0.8624 0.5001 0.0066    0.6292
 baseline_base_rate 2508    1.0333 0.6250 0.0000    0.4737
 baseline_bookmaker 2485    0.8580 0.4972 0.0042    0.6266
blend (odds subset) 2485    0.8685 0.5040 0.0067    0.6262

gap to bookmaker: +0.0106 log-loss
  Behind the closing line, as expected.
```

**These specific numbers came from synthetic data** generated to exercise the pipeline (the sandbox this was built in had no network access to football-data.co.uk). The signal in that data is cleaner than reality, so real log-loss will be higher — expect roughly 0.96–0.99 for a top league, against a bookmaker around 0.95–0.97. Run `scripts/backtest.py` yourself for the real figures and replace this table.

What the shape of it shows: the model comfortably beats base rates, sits slightly behind the closing line, and is well calibrated. Dixon-Coles calibrates better than Elo (ECE 0.005 vs 0.017), which is the expected consequence of modelling scorelines directly.

**We do not expect to beat the bookmaker.** Closing odds aggregate an enormous amount of information and are the hardest benchmark in the field. If the gap ever goes negative, the first hypothesis is a leakage bug, not an edge.

## Data

[football-data.co.uk](https://www.football-data.co.uk/data.php) — free CSVs, no key, no scraping, results plus closing odds from 15+ bookmakers back to 1993.

Two traps in this source, both handled and both tested:

- Dates are `dd/mm/yyyy`. Parsed without `dayfirst=True`, `05/08/2024` becomes 8 May and the whole dataset silently reorders.
- `B365H` is the **opening** price; the closing line is `B365CH`. Only closing odds are stored, so the opening price cannot be used by accident.

Models train on all five leagues pooled and display one. Single-league data is ~380 matches a season; pooled is ~1,750. At single-league volume most apparent improvements are noise, which becomes acute once an agent is proposing features in a loop.

## Layout

```
src/fpp/
  ingest/     football-data.co.uk CSVs, team-name normalization
  models/     elo.py, dixon_coles.py, blend.py
  evaluation/ metrics.py, backtest.py
  api.py      FastAPI: JSON endpoints + one server-rendered page
scripts/      ingest, backtest, predict
docs/decisions/  ADRs — why things are the way they are
tests/        74 tests; model maths checked against closed-form values
```

## Testing

```bash
pytest tests/ -q     # 74 passed
```

Tests assert against known truth, not stored snapshots. Synthetic data is generated from *known* team strengths, so the tests check that the model recovers them — a failure should always be explainable as "the model is now wrong about X", never "a number moved". The Elo update is checked against the closed-form 400-point/10:1 property; the Dixon-Coles `tau` against the paper's definition; the gradient against a numerical one.

## Decisions

See [`docs/decisions/`](docs/decisions/). Some that shaped this:

- [0002](docs/decisions/0002-premier-league-first.md) — Premier League before La Liga, because the FPL API is the only free source with injury and expected-availability data.
- [0003](docs/decisions/0003-sql-for-structured-rag-for-text.md) — RAG is for unstructured text, not for the match database. Embedding rows to retrieve them by cosine similarity converts an exact lookup into an approximate one.
- [0007](docs/decisions/0007-squad-similarity-is-sql.md) — cosine similarity between two one-hot starting XIs is `shared/11`. That's a join and a count, not a vector index.

## Known limitations

- **No forward fixtures.** football-data.co.uk publishes results, not upcoming fixtures, so the landing page currently demonstrates on notable pairings. A fixtures feed is the first item in week 4.
- **No player-level model yet.** Goal probability, ratings, and head-to-head form are weeks 7–12.
- **De-vigging is multiplicative**, which removes margin evenly across outcomes. Real books load more onto longshots, so this slightly overstates unlikely outcomes. Shin's method would be more accurate if the baseline comparison becomes load-bearing.
- **Blend weights are a prior, not fitted.**
- **Promoted teams start at the league average**, so early-season predictions for them are weak.

## Roadmap

| Phase | Scope |
|---|---|
| **1 (done)** | Ingestion, Elo + Dixon-Coles, evaluation harness, API and page |
| 2 | MCP server over the DB; ingestion/entity-resolution agent; maintenance agent on GitHub Actions |
| 3 | RAG over football news and match reports; hybrid retrieval; Ragas evals in CI |
| 4 | Gradient-boosted ensemble and player models, built by an experimentation agent under an evaluator-optimizer loop |
| 5 | Scraper-repair agent, more leagues, deployment |

## Attribution

Results and closing odds from [football-data.co.uk](https://www.football-data.co.uk/). Dixon, M.J. and Coles, S.G. (1997), "Modelling Association Football Scores and Inefficiencies in the Football Betting Market", *JRSS-C* 46(2), 265–280.
