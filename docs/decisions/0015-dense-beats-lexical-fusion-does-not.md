# 0015 — Dense retrieval beats lexical; fusing them does not beat dense

**Status:** accepted
**Date:** week 4 (RAG phase, PLAN weeks 7–9)

## Context

ADR 0014 built BM25 over a 978-chunk Wikipedia corpus and measured it alone —
recall@5 = 11/13, MRR 0.750 — specifically so that whatever dense retrieval added
would be attributable rather than assumed. This adds the dense half and the
Reciprocal Rank Fusion stage that PLAN §4 specifies, and reports what happened.

## Decision

Ship all three modes. **Keep `hybrid` as the CLI default**, despite dense scoring
higher on this question set, and record why the measurement does not yet justify
changing it.

## The measurement

Voyage `voyage-3-large`, 1024 dimensions, same 13 hand-written questions, same
corpus, `recall@5`:

| | lexical | dense | hybrid |
|---|---|---|---|
| recall@5 | 11/13 | **12/13** | 12/13 |
| MRR | 0.750 | **0.885** | 0.808 |

Dense beats lexical on both metrics: +1 recall, +0.135 MRR. It wins exactly
where ADR 0014 predicted it would — the paraphrase cases with no keyword bridge.
"where do they play their home games" moves from rank 4 to rank 2; "record
number of goals scored in a season", which lexical missed entirely, comes back
at rank 1.

**Fusion, however, is worse than dense alone**, and that is the result worth
recording because PLAN §4 assumes otherwise.

## Why fusion lost, precisely

Take "record number of goals scored in a season". Dense ranks the correct
2023–24 season article **1st**. Lexical ranks it **12th**, having filled its top
places with club "Record goalscorers" sections, which share the query's
vocabulary and answer nothing.

RRF at k = 60 then computes:

```
correct 2023-24 article   1/(60+1) + 1/(60+12)  = 0.0164 + 0.0139 = 0.0303
Sunderland "Statistics"   1/(60+3) + 1/(60+4)   = 0.0159 + 0.0156 = 0.0315   <- wins
```

A document both rankers place *moderately* beats a document one ranker places
first and the other places twelfth. That is not a bug in the implementation —
it is precisely the property RRF exists for, the one `test_agreement_beats_
enthusiasm` asserts. Here the enthusiasm happened to be right and the agreement
happened to be wrong.

The general form: **fusion assumes both rankers are independently informative.
When one is systematically wrong on a query class, fusion launders that
wrongness into the output via agreement.** Lexical is not merely weaker on
paraphrase queries; it is confidently wrong, and RRF has no way to know that.

## Why the default does not change

Dense wins on this set. The default stays `hybrid` anyway, for one reason:
**thirteen cases is not enough evidence to overturn a well-supported general
result.** One case is 7.7% of recall. The recall difference between dense and
hybrid is zero; the MRR difference rests on two queries where fusion demoted a
correct dense hit from 1st to 4th, against one where it promoted a correct hit
from 2nd to 1st. Two-against-one is not a finding.

Switching a default on that evidence would be the same error this project keeps
declining to make elsewhere — ADR 0008 froze the models rather than tune against
a table that had already been inspected. The honest position is that the
measurement is suggestive, the sample is too small, and the number is published
either way.

What would settle it: more questions. Fifty would make a one-case swing worth
2% instead of 7.7%. That is cheap to write and is the obvious next step, ahead of
any further retrieval machinery.

## What this says about the rerank stage

PLAN §4's cascade ends with a cross-encoder rerank over the fused candidates,
and this measurement turns that from a step in a plan into a motivated fix. The
failure above is *recoverable at rerank*: the correct passage is in the fused
candidate set at rank 4, and a reranker scoring query-passage pairs directly
would not care that lexical ranked it 12th. Fusion's job is recall into the
candidate pool; ordering that pool is the reranker's.

So the pipeline is not wrong — it is incomplete, and now incomplete for a
demonstrated reason rather than an assumed one.

## Alternatives considered

**Weight the two rankers instead of using RRF.** Rejected on the same grounds
PLAN §4 gives: BM25 scores and cosine similarities have no comparable scale or
fixed range, so any weighting of raw scores is arbitrary. A weighting of *ranks*
is defensible, but tuning that weight on thirteen cases would fit the noise.

**Drop lexical and ship dense only.** Tempting on the numbers, rejected on the
sample size — and lexical still wins the cases it was chosen for: exact names
and terms, where an embedding can drift. PLAN §4 wants both for that reason.

**Tune k in RRF.** Lowering k sharpens the advantage of a first-place finish and
would have fixed both regressions here. That is fitting a constant to two
observations, which is exactly what a thirteen-case eval cannot support.

## Consequences

**Vectors live in SQLite, not pgvector.** ADR 0005 defers Postgres until
pgvector is needed. Cosine over 978 chunks is one 978×1024 matrix multiply, so
it is not needed. Migrating on the assumption of scale that does not exist would
be infrastructure bought ahead of a measurement.

Embedding the corpus took **22 minutes** for 978 chunks, entirely because
Voyage's unbilled free tier allows 3 requests and 10,000 tokens per minute.
Adding a payment method lifts the limits and leaves the 200M free tokens intact,
after which `BATCH` rises and `REQUEST_DELAY` falls to near zero. The embedder
persists per batch and skips work already done, so a run resumes rather than
restarting.

One case, "manager sacked after one win in thirteen games", fails in **all three
modes**. The passage exists in the 2023–24 season article, but it competes with
twenty club histories full of managerial sackings, and neither lexical nor dense
separates the specific from the generic. It is left failing and visible.

The eval's expected-document check is stricter than "a good answer": for
"first team relegated to the Championship" the 2025–26 Relegation section that
hybrid returns is arguably a fine answer, and it is scored as a miss because the
case names the 2023–24 article. That sharpens the comparison between modes on a
fixed target, and it means the absolute recall figures understate usefulness.
