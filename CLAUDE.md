# CLAUDE.md

Project context for Claude Code. Read this before making changes.

## What this is

A football match prediction platform. The **primary goal is to learn and demonstrate agentic AI, RAG, and multi-agent engineering**; football prediction is the domain that makes it concrete. Where the two goals conflict, agent engineering wins.

Current stage: **week 4.** The deterministic vertical slice is complete and frozen (ADR 0008), CI and the `PreToolUse` guardrail are in place (ADR 0009), and the forward-fixture feed landed (ADR 0010). Still no agents, no RAG, no ML. See `docs/PLAN.md` for the full roadmap.

## Non-negotiables

1. **Never train or fit on data from after the match being predicted.** Every backtest is walk-forward. If you touch `evaluation/backtest.py`, this is the invariant to preserve — leakage silently inflates every metric downstream and is the single easiest way to make this project worthless.
2. **Three outcomes, never two.** Home / draw / away. Probabilities sum to 1. Any function returning a "win probability" scalar is a bug.
3. **The bookmaker is the benchmark, not the target.** We expect to lose to de-vigged closing odds. Report the gap honestly; do not tune until we "win", which would mean we have overfit.
4. **Don't claim betting edge** anywhere in code comments, README, or UI.
5. **Every significant decision gets an ADR** in `docs/decisions/`. See §"Decision log" below.

## Layout

```
src/fpp/
  config.py         settings, paths, DB URL
  db.py             SQLAlchemy engine + schema helpers
  schema.sql        table definitions (SQLite/Postgres compatible)
  ingest/
    football_data_uk.py   CSV ingester for football-data.co.uk results
    fixtures.py           upcoming-fixture feed from the same source
    teams.py              team-name normalization across sources
  models/
    elo.py           Elo ratings + fitted Elo->1X2 mapping
    dixon_coles.py   time-weighted Poisson with tau correction
    blend.py         weighted blend + calibration
  evaluation/
    metrics.py       log-loss, Brier, calibration, de-vig
    backtest.py      walk-forward harness
  api.py            FastAPI app
scripts/            CLI entry points (ingest, fixtures, backtest, predict)
tests/              pytest; model math is verified against closed-form values
```

## Conventions

- Python 3.10+. Type hints on public functions. No classes where a function will do.
- Probabilities are always ordered `(home, draw, away)` — as tuples, arrays, and DB columns. Consistency here prevents a whole category of silent bugs.
- Money/odds are decimal odds (2.50), never fractional or American.
- Dates are `date` objects internally; the source CSV uses `dd/mm/yyyy`, which pandas will misparse as US format unless `dayfirst=True` is passed. This has bitten before.
- DB access goes through `db.py`. No raw connections scattered around.

## Data sources (current stage)

Only **football-data.co.uk** so far — free CSVs, no key, no scraping, results plus closing odds from multiple bookmakers, back to 1993. URL pattern: `https://www.football-data.co.uk/mmz4281/{season}/{div}.csv` where season is e.g. `2425` and div is `E0` (Premier League), `SP1` (La Liga), `I1`, `D1`, `F1`.

Column notes that are easy to get wrong:
- `FTHG` / `FTAG` are full-time goals; `FTR` is `H`/`D`/`A`.
- `B365H`/`B365D`/`B365A` are **opening** odds. The closing odds carry a `C`: `B365CH`, `B365CD`, `B365CA`. **Use closing odds** — they are the informed benchmark. `AvgCH`/`AvgCD`/`AvgCA` are the market average and are preferable when present.
- Older seasons are missing many columns. Never assume a column exists.
- Closing odds only exist from 2019/20 onward. The bookmaker benchmark therefore covers part of the history, not all of it.

**Upcoming fixtures** come from `https://www.football-data.co.uk/fixtures.csv` — one file, all divisions, about a week ahead. It is shaped differently from the season files in four ways, each a silent failure rather than an error:

- It is **UTF-8 with a BOM**. Read it as `latin-1` like the season files and the first column is not `Div`, so the division filter matches nothing and the ingest quietly does nothing.
- There is **no season column**. Derive it with `config.season_code`, which must agree with how the source names its results files, or a fixture never reconciles with its own result.
- There are **no closing odds** — the C-suffixed columns are present but empty. Do not store the opening prices that *are* present.
- Fixtures **must be inserted with `ON CONFLICT DO NOTHING`**, never `DO UPDATE`. They share a uniqueness key with results, and the upsert pattern used by the results ingester would write NULLs over finished matches.

Team names differ between the two feeds within this one source (results say `Ath Madrid`, fixtures say `Atl. Madrid`). Unaliased, that splits a club in two and produces a confident, meaningless prediction. `scripts/fixtures.py` warns about teams with no match history; treat that warning as a probable alias gap, not a promoted club, until checked.

**Predictions for future matches are locked once the match date passes** (`db.store_predictions`). Re-running refreshes a fixture while it is still ahead and refuses to touch it afterwards. Do not "fix" this — a prediction that can be rewritten after the result is known is not evidence of anything. See ADR 0010.

Coming later (do not build yet): Fantasy Premier League API for injuries and expected minutes, Understat for xG, FBref for per-90 stats.

## Training data policy

The site displays the **Premier League**, but models train on the **pooled top-5 leagues** (E0, SP1, I1, D1, F1) with league as a feature. A single league is ~380 matches a season; pooled is ~1,750. This matters because at single-league volume, most apparent improvements are noise.

## Testing

`pytest tests/`. Model math is tested against closed-form values, not snapshots — e.g. the Dixon-Coles tau correction and the Elo update are checked against hand-computed results. If you change a model, the test should fail for a *reason you can state*, not because a number moved.

## Decision log

`docs/decisions/` holds one short ADR per significant choice: the decision, alternatives considered, reasoning. This exists because the project goes on a CV and the hardest question about any AI-assisted project is "which parts were your judgment?" — that answer has to be recorded as it happens, not reconstructed later. When you make a non-obvious call, write the ADR.

## Not yet built (don't scaffold ahead)

Agents, MCP server, RAG, vector search, XGBoost, player models, Next.js frontend. Each has a roadmap slot. Building them early creates surface area with nothing to attach to.
