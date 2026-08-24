# 0014 — Lexical retrieval first, measured, before any vectors

**Status:** accepted
**Date:** week 4 (RAG phase, PLAN weeks 7–9)

## Context

ADR 0003 split retrieval: structured facts are queried, unstructured text is retrieved. The query half shipped as the MCP tool layer (ADR 0011). This starts the text half.

PLAN §4 specifies the destination — metadata filter, then hybrid search fusing BM25 with dense vectors via Reciprocal Rank Fusion, then a cross-encoder rerank. It also says, in the same paragraph, that BM25 "is the highest-impact addition to a pure vector pipeline."

That sentence is a claim about how much lexical matching contributes on its own, and it is unverifiable without a number for lexical alone.

## Decision

Build and measure **BM25 over a Wikipedia corpus first**, with metadata pre-filtering, and no embeddings. Dense retrieval is added next and has to beat this on the same question set.

## Reasoning

**Same discipline as ADR 0008.** That ADR froze the deterministic models because the gap to the bookmaker was measured rather than assumed. Adding dense retrieval and a reranker in one step would produce a pipeline that works with no way to say which part earned its place — and "we know when *not* to reach for the fashionable technique" is, per PLAN §12, the judgement this project is meant to demonstrate. A hybrid built without a lexical baseline cannot demonstrate it.

It also defers two real costs until measurement justifies them: the Postgres migration (ADR 0005 says Postgres arrives *when pgvector is needed*, not before) and a second API key for embeddings.

**BM25 is written out rather than imported.** It is a closed-form formula, so the tests check it against values computed by hand — the same standard applied to the Elo update and the Dixon-Coles tau correction. Importing it would leave a load-bearing component nobody here can explain under questioning, which is the failure mode PLAN §14 is written against.

The Lucene IDF variant, `ln(1 + (N − n + 0.5)/(n + 0.5))`, is used instead of the textbook form. The textbook IDF goes negative for a term appearing in most documents, which lets a common word actively push a document *down* the ranking. In a corpus where nearly every document contains "football" and "season", that is a live effect rather than a corner case.

**No stemming.** PLAN §4 wants lexical search precisely for "exact terms like player and manager names", and a stemmer is most aggressive on exactly those proper nouns.

**Wikipedia before news.** News RSS is the richer source for team news, and it evaporates: a retrieval eval scored against a corpus that changes underneath it measures nothing twice. Encyclopedia articles are stable, CC BY-SA (attribution and share-alike, so the licence is stored per document), and available through a documented API. News arrives later under the PLAN §11 constraint — headline, excerpt, link, and our own derived facts only.

### The measurement

23 documents, 978 chunks, ~138,000 words. Thirteen hand-written questions, deliberately mixed: some share vocabulary with the target passage, others are paraphrases with almost no lexical overlap.

```
recall@5   11/13  (85%)
MRR         0.750
random@5   ~0.5%  (floor, for scale)
```

The two failures are the informative part, and both are unfiltered queries where a topically plausible passage from the wrong document outranked the right one:

| query | retrieved instead | wanted |
|---|---|---|
| "record number of goals scored in a season" | Sunderland → *Record goalscorers* | 2023–24 season → "a record 1,246 goals" |
| "manager sacked after one win in thirteen games" | Nottingham Forest → club history | 2023–24 season → Steve Cooper's dismissal |

Neither is a bug. Both are exactly what lexical matching does: strong term overlap, wrong document. That is the gap dense retrieval is supposed to close, and there is now a number attached to it.

The eval also checks its own cases. When a query misses, it verifies the expected text exists in the expected document, and reports a `BROKEN` case rather than scoring a bad expectation as a retrieval failure. An eval that quietly counts its own mistakes as model failures is worse than no eval.

## Alternatives considered

**Build the full hybrid pipeline in one step.** What PLAN describes, and it produces a system with no way to attribute the result to any component. Rejected for the reason above.

**Import `rank_bm25`.** One less thing to write, one more thing that cannot be defended in an interview, and the formula is fifteen lines.

**News RSS or GDELT as the first corpus.** Better content, unstable substrate. Deferred rather than rejected.

**Fixed-window chunking.** Splitting on Wikipedia's own section headings keeps a passage about one thing, because the headings are a real outline written by a person. Long sections are then packed by paragraph, and a single paragraph over the cap is hard-split — without that the cap does nothing, which a test caught.

## Consequences

**A silently wrong document nearly entered the corpus.** Resolving the club "Hull" produced `Hull F.C.` — a **rugby league** club — and it passed a `"football club" in text` check, because English rugby league clubs are also named Football Club. Twenty-nine chunks of rugby were filed under a Premier League team before the title in the output looked wrong. The check now requires a football-club signal *and* the absence of other-sport markers, there is a regression test, and two clubs whose names do not reach the right article by rule (`Hull`, `Ipswich`) have explicit overrides rather than a fuzzy fallback — the same refusal-to-guess as ADR 0012.

This is the second time a plausible-but-wrong entity nearly entered the data silently. The pattern is consistent: the failure never raises, and the output looks fine.

Retrieval is currently unused by anything. The analyst agent that routes between the MCP tools and this corpus is PLAN §6 and arrives after the dense half. `documents` and `chunks` live in SQLite alongside everything else; the pgvector migration is the dense step's problem, not this one's.

The BM25 index is rebuilt per query. At 978 chunks that is milliseconds, and it means the index and corpus can never disagree. It becomes worth caching when a measurement says so.
