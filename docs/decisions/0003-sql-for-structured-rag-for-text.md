# 0003 — SQL/tool-use for structured data, RAG only for text

**Status:** accepted
**Date:** week 1 (implementation lands weeks 4–9)

## Context

Demonstrating RAG is one of the project's stated goals. An early design proposed generating text "cards" from the match database — a team form card, a player season card — embedding those, and retrieving them by vector similarity to answer user questions.

## Decision

Split retrieval by data type. Structured facts are served by purposeful MCP tools over SQL (`get_match_prediction`, `get_player_form`, `get_head_to_head`), with a guarded read-only text-to-SQL escape hatch for open-ended queries. RAG is reserved for genuinely unstructured text: news, injury reports, tactical analysis, Wikipedia match reports.

## Reasoning

The card-embedding design converts an exact lookup into an approximate one. "How many goals has Haaland scored against Arsenal?" has an exact SQL answer; approximate nearest-neighbour search over a generated paragraph is slower, costlier, and can silently return the wrong row. Aggregations, joins, and filters — which most football questions need — are native to SQL and awkward in RAG.

Exposing a few purposeful tools rather than one raw `execute_sql` is deliberate: constrained tools are safer, more reliable for the model to call correctly, and force explicit thought about tool ergonomics, which is the part of agentic engineering that actually determines whether the system works.

## Alternatives considered

**RAG over generated cards (the original plan).** Would have demonstrated the mechanics of RAG while demonstrating poor judgment about when it applies. Rejected.

**Text-to-SQL only, no RAG.** Simpler, but discards genuinely unstructured sources that carry real predictive context — injury news, expected lineups — and drops a stated learning goal. Rejected.

## Consequences

Two retrieval paths to build and evaluate instead of one, and the analyst agent needs routing logic to choose between them. That routing decision is itself worth evaluating and is a more interesting thing to show than a single path would be.
