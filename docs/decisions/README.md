# Decision log

One file per significant decision: what was decided, what else was considered, and why.

This exists for a specific reason. The project is AI-assisted, and the hardest question anyone will ask about it is "which parts were your judgment?" That question is answerable only if the reasoning is recorded when it happens — reconstructing it months later produces vague answers that sound exactly like someone who didn't make the decisions.

Write an ADR when a choice is non-obvious, has a real alternative, or would confuse a reader six months from now. Skip it for obvious calls.

Format: numbered, short, present tense. Status is `accepted`, `superseded by NNNN`, or `rejected`. An ADR is immutable once accepted — if you change your mind, write a new one that supersedes it. Superseded decisions stay in the log; the trail of reversals is often the most interesting part.

| # | Decision | Status |
|---|---|---|
| [0001](0001-record-decisions.md) | Record architecture decisions | accepted |
| [0002](0002-premier-league-first.md) | Premier League first, La Liga second | accepted |
| [0003](0003-sql-for-structured-rag-for-text.md) | SQL/tool-use for structured data, RAG only for text | accepted |
| [0004](0004-pooled-multi-league-training.md) | Train on pooled top-5 leagues, display one | accepted |
| [0005](0005-sqlite-before-postgres.md) | SQLite now, Postgres when pgvector is needed | accepted |
| [0006](0006-defer-js-frontend.md) | Server-rendered HTML now, defer the JS frontend | accepted |
| [0007](0007-squad-similarity-is-sql.md) | Squad similarity is SQL, not a vector index | accepted |
| [0008](0008-freeze-the-deterministic-baseline.md) | Freeze the deterministic baseline; don't tune the gap away | accepted |
| [0009](0009-agent-guardrails-are-structural.md) | Agent guardrails are structural, not prompted | accepted |
| [0010](0010-forward-fixtures-and-locked-predictions.md) | Forward fixtures from the same source; predictions locked at kick-off | accepted |
| [0011](0011-mcp-tools-over-raw-sql.md) | Purposeful MCP tools, and read-only enforced below the code | accepted |
| [0012](0012-resolution-refuses-to-guess.md) | Entity resolution refuses to guess; availability is a time series | accepted |
| [0013](0013-entity-resolution-agent-proposes.md) | The entity-resolution agent proposes; it does not write | accepted |
