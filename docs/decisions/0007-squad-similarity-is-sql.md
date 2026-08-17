# 0007 — Squad similarity is SQL, not a vector index

**Status:** accepted
**Date:** week 1 (implementation lands week 10+)

## Context

A planned feature weights historical head-to-head matches by how closely the lineup resembles today's expected XI, so results from a squad that has since turned over count less. An early design proposed encoding each starting XI as a one-hot vector over all player IDs and using pgvector cosine similarity to compare them.

## Decision

Compute it in SQL. No embeddings, no vector index.

## Reasoning

Cosine similarity between two one-hot starting-XI vectors reduces to `|shared players| / 11`.

Each vector has exactly eleven 1s, so the dot product is the size of the intersection and both norms are √11, giving `|A ∩ B| / (√11 · √11)`. That is a join and a count. (Verified numerically — see `tests/test_squad_similarity.py` when the feature lands.)

Minutes-weighting, where a full 90 counts more than a late substitute appearance, makes it a weighted sum. Still SQL.

## Alternatives considered

**pgvector with one-hot vectors.** The original proposal. Solves arithmetic with infrastructure, and adds an index, a dimension choice, and an approximate-search error term to an exact integer computation. Rejected.

**Learned player embeddings instead of one-hot.** Genuinely different — similarity would capture playing style rather than identity, so a like-for-like replacement could score as similar. Interesting, and vector search would be justified. Rejected *for now* as a much larger project with no obvious training signal; worth revisiting if the simple version shows promise.

## Consequences

pgvector remains in the stack for the RAG text index (see 0003), just not for this. The feature is expected to have thin effective sample size, since squad turnover means few historical matches score highly similar — it should be evaluated with the expectation that it may show no significant improvement, and reported as such if so.
