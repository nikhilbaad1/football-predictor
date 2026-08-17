# Football Prediction Platform — Architecture & Research Plan

*Rewritten from scratch. This version corrects several things the earlier draft got wrong — the RAG design, the squad-similarity math, and the overall scope. Section 12 lists what changed and why, so you can see the reasoning rather than just the conclusion.*

## 1. What this project actually is

The primary goal is to learn and demonstrate agentic AI, RAG, and multi-agent engineering. Football prediction is the domain that makes that concrete — a good choice, because it has messy multi-source data, genuine unstructured text, and outputs that can be objectively scored against reality. That last property matters more than it sounds: most agent portfolio projects produce output nobody can grade. This one can be graded by the season.

That reframing changes what "done" means. The thing you're shipping is a repository with a documented multi-agent system, a real evaluation suite with real numbers, and a modest live site — not a comprehensive football statistics product. Where the two goals conflict, the agent engineering wins.

## 2. Honest expectations, stated up front

**Three outcomes, not two.** Your original example gave Real Madrid 0.6 and Barcelona 0.4. Football has home win, draw, and away win — roughly a quarter of top-league matches are draws — so every output here is a triple, e.g. 0.47 / 0.28 / 0.25.

**You will not beat the bookmakers, and shouldn't design as if you will.** Closing odds aggregate enormous amounts of information and are the hardest benchmark in the field. The academic literature is roughly: careful models get *near* bookmaker calibration, occasional papers claim positive ROI, and those claims are fragile once you account for margin, execution, and multiple-testing. Target being *well-calibrated and honest about it* — that is both achievable and a better story than an unverifiable claim of edge.

**The sample size is small and this is the binding constraint.** The Premier League plays 380 matches a season. Ten seasons is ~3,800 rows, of which maybe 400 form a held-out validation set. On a set that size, a 1% log-loss "improvement" is usually noise. Two consequences run through the whole design: train on the top-5 leagues pooled (~1,750 matches a season, so ~17,000 over a decade) even if the site only displays one, and build the evaluation harness to *reject* most proposals rather than to find winners.

## 3. Data sources

Tiered by legal risk, because that determines what you can safely put on a public site.

**Tier 1 — public domain or explicitly free, zero risk. Start here.**

| Source | What it gives | Access |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/data.php) | Results + closing odds from 15+ bookmakers, 20+ leagues, back to 1993 | CSV download, no key |
| [OpenFootball](https://github.com/openfootball/football.json) | Fixtures/results for major leagues, JSON | GitHub, public domain, no restrictions |
| [Fantasy Premier League API](https://fantasy.premierleague.com/api/bootstrap-static/) | Official Premier League: per-player stats, **injury/availability status and "chance of playing next round"**, team strength, fixture difficulty | Public JSON, no key |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Full event data with x/y coordinates, lineups, 360 data — selected competitions only | GitHub, attribution required |
| [ClubElo](http://clubelo.com/System) | Pre-computed club Elo, full history | Free REST/CSV |
| Wikipedia / Wikidata | Season articles, match reports, squad history | Wikipedia CC BY-SA (attribution + share-alike); Wikidata CC0 |

**Tier 2 — free but no explicit reuse grant. Fine for a personal project; cache aggressively.**

[Understat](https://understat.com) for shot-level xG/xA, [FBref](https://fbref.com) for per-90 player stats and match reports. The [`soccerdata`](https://github.com/probberechts/soccerdata) Python package wraps both plus ClubElo and football-data.co.uk behind one interface with local caching.

**Tier 3 — avoid.** Transfermarkt's ToS actively discourages scraping. The earlier draft listed it; drop it. Market values are a nice-to-have, not worth the risk on a public site.

The FPL API is the single highest-value find here, and it changes the league recommendation. It is official, free, requires no key or scraping, and its availability flags directly solve the expected-minutes problem that §5.3 otherwise has to hand-wave. Nothing equivalent exists free for La Liga. So: **build the Premier League first**, pool the other top-5 leagues into training data, and add La Liga to the display once the pipeline works — at which point your Clásico example works too.

One rule matters more than source selection: fetch once, store permanently, never re-fetch what you already have. This keeps request volume respectful and means a source breaking doesn't take your site down.

## 4. Retrieval architecture — the part the earlier draft got wrong

The earlier plan proposed RAG over text "cards" generated from your own database. That was bad advice and worth correcting explicitly, because getting this distinction right is itself a demonstrable skill.

If a fact lives in Postgres and you know which match you're asking about, retrieval should be a **query**, not a vector search. Embedding a row and retrieving it by cosine similarity converts an exact lookup into a lossy, slower, more expensive one. "How many goals has Haaland scored against Arsenal?" has an exact SQL answer; approximate nearest-neighbour is strictly worse at producing it. Structured questions also frequently need aggregation, joins, and filters — things RAG handles badly and SQL handles natively.

So the system splits cleanly, and the agent decides which path a question needs:

**Structured facts → tool use.** A well-designed MCP server over Postgres, exposing a small number of purposeful tools (`get_match_prediction`, `get_player_form`, `get_head_to_head`) rather than one raw `execute_sql`. Constrained tools are safer, more reliable, and force you to think about tool ergonomics — which is the genuinely hard and underrated part of agentic engineering. Add a guarded text-to-SQL escape hatch (read-only role, statement timeout, row limit) for open-ended questions the fixed tools don't cover.

**Unstructured text → RAG.** This is where retrieval earns its place, over text that has no schema: pre-match news and press conferences, injury and team-news reports, tactical analysis, Wikipedia match reports and season summaries. Sources: RSS feeds from major football outlets (published for syndication — store the headline, a short excerpt, the link, and your own derived facts; do not republish full article text), [GDELT](https://www.gdeltproject.org/) for news at scale with no licensing cost, and Wikipedia under CC BY-SA.

Retrieval pipeline, following current practice: filter by metadata first (these two teams, this date window), then hybrid search — BM25 for exact terms like player and manager names, dense vectors for semantics — fused with Reciprocal Rank Fusion, then a cross-encoder rerank on the top candidates. Hybrid consistently outperforms dense-only; BM25 is the highest-impact addition to a pure vector pipeline, and the entire retrieval cascade costs milliseconds against an LLM call that costs seconds.

Embeddings via Voyage AI `voyage-3-large` (Anthropic's recommended pairing for Claude pipelines), or self-hosted BGE-M3 if you want the stack at exactly zero marginal cost. The vector index lives in pgvector inside the existing Postgres — no separate vector database.

## 5. Prediction methodology

### 5.1 Match outcome

Three layers, blended:

**Elo** — running team strength, updated after each match with a home-advantage term. ClubElo publishes this; compute your own if you want control of the update constant. Cheap, robust, no scoreline detail.

**Dixon-Coles** — time-weighted Poisson regression estimating attack and defence strength per team, producing a full scoreline probability matrix that you sum into 1X2. Its correction factor fixes plain Poisson's known mis-estimation of low scores (0-0, 1-0, 1-1); published backtests put it around 15% better than plain Poisson.

**Gradient-boosted ensemble** — XGBoost or similar over Elo difference, rolling form, xG-based form, rest days, and head-to-head. Trained on pooled top-5-league data with league as a feature, for the sample-size reasons in §2.

Blend, then calibrate with isotonic regression so a stated 60% actually occurs about 60% of the time. Score with log-loss and Brier. Benchmark against margin-removed closing odds from football-data.co.uk — not to beat it, but to know how far off you are.

### 5.2 Player performance rating

WhoScored and FotMob build theirs from proprietary feeds — 200+ and 300+ per-match event stats respectively, both anchored at 6.0 on a 10-point scale. Free data gives you roughly 30–40 per-90 metrics instead. Build a transparent composite with position-specific weights (forwards weighted toward xG and shot quality, defenders toward defensive actions and progressive passing), normalize to 0–10, and publish the weights. Label it as your own metric, not a WhoScored reconstruction. Aggregate into rolling form (last 5–10 matches, recency-weighted) and a per-season career trajectory.

Goalkeepers need a separate basis — post-shot xG minus goals conceded, i.e. shot-stopping above expectation — since outfield metrics are meaningless for them. Worth deciding now rather than retrofitting.

### 5.3 Player goal probability

Poisson shot process: λ = expected shots × conversion rate × opponent defensive adjustment × expected minutes share. Probability of scoring at least once is `1 − e^(−λ)`.

Two details that matter. Conversion rate needs empirical-Bayes shrinkage toward position and league averages — a player at 3 goals from 4 shots is not a 75% finisher, and taking that at face value is the single most common way these models embarrass themselves. And expected minutes is a real input, not a rounding error: a 20-minute substitute has roughly a fifth of a starter's exposure. The FPL API's availability flags feed this directly, which is a large part of why §3 recommends the Premier League.

### 5.4 Head-to-head form

Same model, restricted to matches against this opponent. Sample sizes are tiny — often 10–20 career appearances — so this must be blended with general form rather than shown standalone, weighted so that a small head-to-head sample collapses toward the general estimate. Fit the blend weight by backtesting.

### 5.5 Squad similarity — corrected

Your idea: weight historical meetings by how much the lineup resembles today's expected XI, so a result from a squad that has since turned over counts less. Sound reasoning, and worth building.

The earlier draft proposed pgvector cosine similarity for this, which was over-engineering. Cosine similarity between two one-hot starting-XI vectors reduces to `|shared players| / 11` — because each vector has exactly eleven 1s, the dot product is the size of the intersection and both norms are √11. It's a join and a count in SQL. No embeddings, no vector index. Minutes-weighting adds a weighted sum and is still plain SQL.

Use the result to re-weight head-to-head history feeding §5.1 and §5.4, and treat it as one engineered feature among many. Be prepared for it to show no measurable improvement — squad turnover means few historical matches score highly similar, so the effective sample is thin. If the eval says it doesn't help, report that. A rejected hypothesis, honestly measured, is a better portfolio artifact than a feature you kept because you liked the idea.

## 6. Agent architecture

Two Anthropic patterns from [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) carry the design: orchestrator-workers for routing, evaluator-optimizer for the experimentation loop.

| Agent | Job | Why an agent (not a script) |
|---|---|---|
| **Ingestion & entity resolution** | Reconcile "Cristiano Ronaldo" across sources with different IDs, spellings, and transliterations; flag anomalies | Fuzzy judgement over messy real-world identity — brittle to hard-code, natural for an LLM |
| **Scraper repair** | When FBref changes markup and a parser breaks, diagnose from the failure and the new HTML, fix, open a PR | The most compelling demo in the project: self-healing infrastructure, and genuinely hard to do procedurally |
| **Experimentation** | Propose features from football-domain reasoning, backtest, report honestly | Domain-reasoned hypotheses are something an LLM is actually good at (see below) |
| **Analyst** | Answer questions by routing between MCP tools (structured) and RAG (text); write per-match explanations | The §4 split in action — deciding which retrieval path a question needs |
| **Maintenance** | Nightly health checks, CI failures, `@claude` mentions; fix and open a PR | Runs on `anthropics/claude-code-action` — real Claude Code in a GitHub runner |
| **Orchestrator** | Decompose triggers, delegate, synthesize; holds no write access itself | Subtasks aren't fixed in advance, which is what distinguishes orchestrator-workers from a pipeline |

**On the experimentation agent, more precisely than the earlier draft.** I previously implied hyperparameter search is beneath an LLM; that overstated it — recent work (Kochnev et al., ICCV 2025 workshops) finds LLM-driven HPO can match Tree-structured Parzen Estimator quality while converging faster. The real point is different: with ~400 validation matches, your bottleneck is *evaluation reliability*, not search efficiency, so raw HPO speed buys you little. The valuable contribution is the agent proposing features with domain reasoning — "try days since last match, fixture congestion should degrade performance" — and then a harness rigorous enough to throw most of them out. Use Optuna for the numeric search underneath and let the agent reason about *what to try*, which plays to each tool's strength.

**Autonomy model.** Every agent that touches code or promotes a model stops at a pull request; merging is yours. This is enforced structurally, not by prompt: a `PreToolUse` hook blocks pushes and merges to `main` and blocks destructive SQL outright, regardless of what the agent concludes it should do. Prompted-only restrictions are known to fail under pressure; the hook is what actually holds. Layered underneath: read-only database roles for the analyst agent, sandboxed bash, and a scoped GitHub token.

## 7. Evaluation — the differentiating piece

Most agent portfolio projects generate plausible output and stop. Measuring whether it's *correct* is what reads as engineering, so treat this as a first-class component rather than a phase-5 nicety.

**Prediction quality:** log-loss, Brier score, calibration curves, versus two baselines — always-predict-base-rates, and margin-removed closing odds.

**RAG quality:** [Ragas](https://docs.ragas.io/) in CI — faithfulness, answer relevance, context precision and recall — against a hand-written set of ~50 question/answer pairs. Run it whenever a prompt or retrieval parameter changes.

**Agent quality:** a fixed set of tasks with known-good outcomes (a deliberately broken parser the scraper agent should fix; entity-resolution cases with known answers). Ragas also ships tool-call accuracy and agent-goal-accuracy metrics, so the same harness covers both the RAG and agent sides rather than needing two.

**Statistical honesty:** significance testing on model comparisons, a validation set the experimentation agent never sees during search, and a logged count of proposals rejected. That last number is a feature of the writeup, not an embarrassment.

## 8. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python | Data, ML, and agents in one language |
| API | FastAPI | No serialization boundary with the model code |
| Database | PostgreSQL + pgvector | One database: relational stats, JSONB events, and the *text* vector index (not squad similarity — see §5.5) |
| Frontend | Next.js | Server-rendered, fine for a stats site |
| Agents | Claude Agent SDK (backend agents) + `anthropics/claude-code-action` (maintenance) | Claude-native throughout |
| Tools | Custom MCP server over Postgres | The purposeful-tools layer from §4 |
| Experiment tracking | MLflow, or plain files to start | Don't stand up infrastructure before you have experiments |
| Evaluation | Ragas + custom harness | §7 |

**Dropped from the earlier draft:** Redis, which at hobby traffic adds an operational dependency for no benefit — a materialized predictions table in Postgres is your cache. Also dropped: pgvector for squad similarity (§5.5), and Transfermarkt (§3).

```
Tier-1 sources ──┐
Tier-2 sources ──┼──> Ingestion agent ──> PostgreSQL ──┬──> MCP tools ──┐
News RSS/GDELT ──┘                       + pgvector ───┴──> RAG ────────┼──> Analyst agent
                                              │                          │
                          Model layer (Elo, Dixon-Coles, XGBoost)        │
                                              │                          v
                                              └──────────> FastAPI ──> Next.js

    Experimentation agent ──> Optuna + MLflow ──> PR ──┐
    Scraper-repair agent ─────────────────────> PR ──┼──> you merge ──> production
    Maintenance agent ────────────────────────> PR ──┘
```

## 9. Roadmap

**Weeks 1–3 — one thin vertical slice, end to end.** Ingest football-data.co.uk CSVs for the Premier League. Fit Elo plus Dixon-Coles. Build the evaluation harness and score it against closing odds. Ship a single page showing next weekend's fixtures with three-way probabilities. No agents, no RAG, no ML. This exists so that everything afterward has something real to attach to — and if the project stalls here, you still have a working, honest predictor. Start the decision log (§14) in week 1, not retroactively.

**Weeks 4–6 — first agents.** MCP server with three or four purposeful tools. Ingestion and entity-resolution agent as you add FPL and Understat. Maintenance agent wired up via GitHub Action, proven on a trivial PR before it touches anything real. Agent eval harness.

**Weeks 7–9 — RAG.** News/RSS and Wikipedia ingestion, hybrid retrieval with reranking, analyst agent routing between tools and RAG, Ragas in CI. This is the section a reviewer will read most closely.

**Weeks 10–12 — the ML and experimentation loop.** Pooled multi-league training, the XGBoost layer, player models, and the experimentation agent proposing and mostly failing to prove features. Squad similarity as one candidate feature among them.

**Beyond.** Scraper-repair agent, more leagues including La Liga, richer player views, public deployment.

## 10. Building it with Claude Code

Two distinct things share the word "agent" here: the subagents helping you *write* the code interactively, and the production roster in §6 that runs after you ship. Keep them separate mentally; the maintenance agent is roughly the dev-time review agent promoted to run unattended.

Start with a `CLAUDE.md` documenting the schema, per-source quirks (FPL field meanings, Understat's JSON-in-script-tag format, FBref rate limits), and the hook-enforced guardrails so interactive sessions respect the same boundaries as production agents. Connect the Postgres MCP server early — it's both your dev tool and a production component, so building it first means you dogfood your own tool design.

Split `.claude/agents/*.md` by concern: data-pipeline, modeling, agent-systems, frontend, and a review agent checking against schema and conventions. Wire up the GitHub Action in week 4 pointed at something trivial, purely to prove the PR loop before trusting it with real work.

## 11. Legal and responsible use

Attribution for StatsBomb, Wikipedia (CC BY-SA also carries share-alike obligations for derived text), and any other source whose terms require it. For news: store links, headlines, short excerpts, and your own derived facts — do not republish article bodies.

Add a clear disclaimer that predictions are probabilistic estimates, not guarantees, and not betting advice. Gambling-adjacent content is regulated very differently across jurisdictions, and the rules change depending on whether money changes hands on your site. If you ever monetize or add affiliate links, get actual legal advice — I'm not a lawyer and this document isn't a substitute for one.

## 12. What changed from the previous draft, and why

**RAG over self-generated cards → SQL/tool-use for structured data, RAG for real text.** The original design applied approximate retrieval to data you could query exactly. The corrected split is both technically better and a stronger demonstration, since knowing when *not* to reach for RAG is the harder judgement.

**pgvector for squad similarity → SQL.** Cosine over one-hot XIs is `shared players / 11`. The vector index was solving arithmetic with infrastructure.

**La Liga → Premier League first.** The FPL API is official, free, unscraped, and carries injury and expected-availability data that nothing free covers for La Liga. La Liga comes second, at which point your Clásico example works.

**Single-league training → pooled top-5 training.** 3,800 matches is too few to learn much; ~17,000 is workable. Display stays single-league.

**Scope cut.** Redis and Transfermarkt dropped. The roadmap now leads with a three-week shippable slice rather than a foundation phase that produces nothing usable.

**Evaluation promoted to its own section** and moved early in the roadmap, because it's the most differentiating and most commonly skipped part of an agentic portfolio.

**Softened the claim about LLMs and hyperparameter tuning** — recent work shows LLM-driven HPO can match TPE. The argument for domain-reasoned feature proposals over raw search stands, but for a different reason than I first gave.

## 13. Confirmed decisions

All open questions from the previous draft are now settled:

- **GitHub** — confirmed. The maintenance agent's PR workflow and the `PreToolUse` guardrails in §6 are good to build as specified.
- **Voyage AI** — confirmed. `voyage-3-large` it is; BGE-M3 stays documented as a fallback but isn't the plan.
- **Premier League first** — confirmed. FPL API data flows into §5.3's expected-minutes input from day one. La Liga follows.
- **CV and interviews** — confirmed, and this one has design consequences. See §14.

## 14. Building it to survive an interview

This is now a stated goal, not a side effect, so it belongs in the plan. The constraint it adds: every significant piece must be something you can *defend under questioning*, which is a higher bar than "works."

**Expect the 2026 version of the hardest question: "how much of this did you write?"** Deflecting is worse than answering. The strong answer is specific about judgment rather than typing — which decisions you made, what you rejected, and why. That answer only exists if you record it as you go, so keep a lightweight decision log: one short markdown file per significant choice, stating the decision, the alternatives, and the reasoning. `docs/decisions/` with a dozen entries is a stronger artifact than most code.

Three decisions from this plan are already good interview answers, and it's worth being able to explain each cold:

- **Why SQL for structured data and RAG only for text.** You considered RAG over the whole database, worked out that it converts exact lookups into approximate ones, and split the retrieval paths. Demonstrating you know when *not* to use the fashionable technique is more convincing than using it everywhere.
- **Why the model doesn't beat the bookmakers.** Being able to say "closing odds are near-efficient, I benchmark against them and sit slightly worse, here's my calibration curve" signals statistical maturity. Claiming edge signals the opposite to anyone who knows the field.
- **How you stopped the experimentation agent overfitting.** A validation set the agent never sees during search, significance testing, and a logged count of rejected proposals.

**Build two things specifically for legibility.** A README carrying the architecture diagram, an honest results table including where you lose to the baseline, and a candid limitations section. And a short screen recording of the scraper-repair agent fixing a deliberately broken parser — the most demo-able thing in the project, and the hardest to dismiss as prompting.

**Keep the negative results visible.** "I tested squad similarity and it produced no significant improvement" is a better answer than a feature you kept because the idea appealed to you. Interviewers who have shipped models recognize the difference immediately.

**One thing to avoid:** don't oversell the player rating as equivalent to WhoScored or FotMob. Yours is a transparent composite over ~30 free per-90 metrics; theirs run on 200–300 proprietary event stats. Saying so plainly costs nothing and protects your credibility on everything else you claim.
