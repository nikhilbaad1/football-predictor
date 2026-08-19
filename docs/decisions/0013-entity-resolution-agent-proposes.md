# 0013 — The entity-resolution agent proposes; it does not write

**Status:** accepted
**Date:** week 4

## Context

ADR 0012 built a deterministic resolver that settles what a rule can justify and refuses the rest. On the 20 FPL team names it scored **19 resolved, zero wrong**, leaving one open question: `Coventry City`, whose nearest known names are `Leicester City`, `Manchester City` and `Norwich City` — three different clubs in three different cities.

That refusal is the correct answer for a rule and a useless answer for a pipeline. Something has to decide, and the deciding needs knowledge of which football clubs exist rather than a better string metric. This is the project's first LLM call.

## Decision

An agent (`agents/resolve_agent.py`, Claude Opus 5) decides the names the resolver refuses. It **proposes**; `scripts/resolve.py` writes only under `--apply`. Its output is **validated against the schema** before anyone acts on it, and **`uncertain` is an allowed answer**.

## Reasoning

**Proposing rather than writing, because the two errors are not symmetric.** Reporting an existing club as new creates a duplicate row that someone notices and merges. Matching one club to a *different* club fuses two histories, and every number computed afterwards is wrong with nothing indicating it. The second failure is the one this whole area of the codebase exists to prevent, so a model does not get to commit it unreviewed. The asymmetry is also stated in the system prompt, which instructs the agent to prefer `new_entity` or `uncertain` when the evidence does not clearly favour one existing club.

**The output is validated, not trusted.** A `matched` decision naming a club that is not in the known list is downgraded to `uncertain` rather than stored; a `new_entity` decision carrying a target has it dropped. The model is a source of judgement about football, not a source of truth about this schema — and `apply_resolution` independently refuses to link a name to a team that does not exist. Three separate checks, because the cost of the one failure is unbounded.

**`uncertain` is a first-class answer.** An agent forced to choose between "matched" and "new" will guess when it does not know, and a guess here is precisely the failure being guarded against. Escalation is cheaper than corruption.

**The model is only asked when a rule cannot answer.** `resolve_with_agent` runs the deterministic resolver first and returns immediately if it succeeds. Nineteen of twenty FPL names cost nothing and stay deterministic; the agent handles the residue. This also keeps the agent's contribution measurable — it is scored only on cases the baseline fails.

### The measurement

`scripts/eval_resolver.py` scores the agent on ten names, every one of which the deterministic resolver refuses. Expected answers are facts about real clubs, checked against the 34 English clubs in the database.

```
correct        10/10
WRONG MATCHES  0      <- the number that decides this
missed         0
cost           $0.0410

baseline: refuses all 10, so 0 correct and 0 wrong matches.
VERDICT: ship.
```

Accuracy is not the headline; **wrong matches** is. The baseline scores zero wrong matches trivially, by refusing everything, so any wrong match would make the agent strictly worse where it counts regardless of how many others it got right. The eval exits non-zero on a single wrong match and says so.

The set includes a deliberate trap: `Sheffield Wednesday`, where the database holds `Sheffield United`. Same city, different club, high string similarity — exactly the shape of mistake a similarity threshold makes. The agent answered `new_entity`, reasoning that the similar names are clubs from different cities.

On the one real case it decided `Coventry City` is a new club at 0.95 confidence, which is correct — it was promoted to the Premier League for this season.

## Alternatives considered

**Let the agent write directly.** Fewer steps and it removes the human check on the one operation with unbounded, silent downside. Rejected now; revisit if the eval set grows large enough to justify trusting a confidence threshold, which it is not at ten cases.

**Force a binary decision — matched or new.** Simpler output schema, and it converts "I don't know" into a guess. Rejected.

**A cheaper model.** The whole task is knowing that Coventry City is a real club distinct from Leicester City. That is the capability being bought, and a weaker model is likelier to make exactly the merge error the design prevents. At roughly a cent per name and one pending name, the saving is not worth reasoning about.

**Score the agent in the test suite.** Rejected: CI has no API key, and a unit test whose result depends on a model call measures the model, not the code. The tests fake the client and cover everything around the call; scoring the model is a separate, deliberate, paid step.

## Consequences

The full loop is closed. The agent decided `Coventry City` was new, `--apply` created the club, and re-running the FPL ingest brought in the 32 players that had been withheld while its identity was unsettled — 560 players became 595, and the pending queue is empty.

**Applying a decision exposed a bug worth recording.** Creating the club meant the next ingest resolved the same name by exact match and overwrote `agent:resolve-agent-v0.1` with `exact`, erasing the record that a judgement had been made at all. The upsert in `_record_resolution` now carries `WHERE name_resolutions.decision IS NULL`, so a settled decision is never replaced by a later deterministic one. Provenance is the point of that table; losing it would have made "which names did the agent decide, and was it right?" unanswerable.

Cost accounting includes cached tokens. `usage.input_tokens` excludes them, and the club roster — most of the prompt — is cached, so counting only uncached input would have understated every run.

`anthropic>=0.123,<1` is a new runtime dependency, upper-bounded because the SDK is pre-1.0. `ANTHROPIC_API_KEY` is required only for the agent; ingest, backtest, predict, fixtures, fpl and the MCP server all run without one.

There is still no autonomous agent. This one runs when a person runs it, and writes only when a person passes `--apply`. The maintenance agent that opens pull requests unattended (PLAN §6) needs branch protection settled first.
