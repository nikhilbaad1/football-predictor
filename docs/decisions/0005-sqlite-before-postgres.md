# 0005 — SQLite now, Postgres when pgvector is needed

**Status:** accepted
**Date:** week 1

## Context

The target architecture is PostgreSQL with pgvector, which the RAG layer needs in weeks 7–9. The week 1–3 slice has no vector search and a dataset small enough to fit comfortably in memory.

## Decision

Default to SQLite via a `DATABASE_URL` setting. Access the database through SQLAlchemy Core so the switch to Postgres is a connection-string change. Move when pgvector is actually required.

## Reasoning

Standing up Postgres in week 1 adds setup friction — a container, credentials, a running service — in exchange for capabilities nothing uses yet. The schema is plain relational SQL that works unchanged on both engines.

This mirrors the same rule applied to MLflow elsewhere: don't stand up infrastructure before there's a workload for it. It's also honest about scale — 17,500 rows is not a database problem.

## Alternatives considered

**Postgres from day one.** Avoids a migration and matches production exactly. A reasonable call, and the cost of the migration later is the main argument for it. Rejected because the friction lands in the phase most likely to stall, and SQLAlchemy makes the switch cheap.

**Parquet/CSV files, no database.** Adequate for the models alone, but the API and later agent tooling both want query access. Rejected.

## Consequences

One migration in week 7, covering a handful of tables with no vendor-specific SQL. Two things to watch until then: SQLite's permissive type affinity will accept data Postgres would reject, and it has no native `DATE` type, so dates round-trip as ISO strings. Both are handled in `db.py`, and the schema avoids SQLite-only syntax so the migration stays mechanical.
