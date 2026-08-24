# 0016 — The analyst agent routes between sources, and cites what it used

**Status:** accepted
**Date:** week 4 (RAG phase, PLAN weeks 7–9)

## Context

ADR 0003 split retrieval in two: exact facts are queried, unstructured text is
retrieved. Both halves now exist — five purposeful database tools (ADR 0011) and
a hybrid text retriever over 978 Wikipedia chunks (ADRs 0014, 0015). Nothing had
yet used them together.

`docs/PLAN.md` §6 calls this the analyst agent and describes it as "the §4 split
in action — deciding which retrieval path a question needs". That is the whole
job: a question with an exact answer sent to the text corpus comes back with a
fluent passage that mentions numbers instead of the number, and a question about
history sent to SQL comes back empty.

## Decision

An agent (`agents/analyst.py`, Claude Opus 5) answering questions over both
sources, choosing per question. It **reads only**, **cites its source**, and is
scored on **routing**, not on answer quality.

## Reasoning

**It reuses `mcp_server/tools.py` as plain Python rather than speaking MCP.**
That is the payoff for the rule in `CLAUDE.md` that the tool logic imports
nothing from `mcp`: the same five tools serve an MCP client over stdio and an
in-process agent, and this module needs no server, no subprocess and no protocol
to be tested. Had the logic lived in the protocol adapter, this agent would have
needed a second copy of it.

**The prompt forbids answering from the model's own knowledge.** Claude knows a
great deal about football, and a model that already knows who won the 2023–24
title will answer without checking — at which point the retrieval layer this
project spent two ADRs measuring is decoration, and the answers are unfalsifiable.
Every factual claim must come from a tool result, and "the tools don't answer
this" is an allowed answer.

**It cites.** Retrieved text names the article; structured results say "from the
match database". Without that, a correct answer and a hallucinated one are
indistinguishable to the reader, which defeats the point of having two auditable
sources.

**It cannot write.** Every tool reads, and the connection underneath is opened
read-only (ADR 0011). The `PreToolUse` hook cannot see an in-process tool call
any more than it can see an MCP one, so the guarantee has to live in the
connection — the same argument, now covering a second caller.

### The measurement

`scripts/eval_analyst.py`, 12 questions, **12/12 routed correctly, $0.56**.

The interesting case is the one needing both: *"What is Arsenal's record against
Chelsea, and what is the history behind the rivalry?"* — answered with
`get_head_to_head` **and** `search_text`. The record is a count; the history is
prose; the agent split the question along exactly the seam ADR 0003 describes.

**Why this ground truth is better than the retrieval eval's.** ADR 0015 admitted
that the retrieval relevance labels were authored by the person who built the
retriever, after reading the corpus, which makes them contestable. These labels
are not judgement calls. A question is `structured` when the fact it asks for has
a column, and `text` when it does not. "How many times has X beaten Y" is a COUNT
over `matches.result`. "What is the club's nickname" has no column anywhere in
`schema.sql`. The label is checkable against the schema rather than against an
opinion, which is a materially stronger claim than the retrieval eval can make.

**What it does not measure: whether the answers are correct.** Routing is
necessary and not sufficient. Answer quality needs its own harness — PLAN §7
wants Ragas for faithfulness and context precision — and nothing here claims it.

## Alternatives considered

**Have the agent talk to the MCP server over stdio.** More faithful to how a
production client would connect, and it buys a subprocess, a protocol round trip
and a much harder test setup for no behavioural difference. The MCP server still
exists and is still the interface for external clients; this is a second caller
of the same logic.

**Let the model answer from its own knowledge when tools come back empty.** It
would produce better-looking answers and destroy the property that makes them
checkable. Rejected.

**Score answer correctness instead of routing.** The right eventual measure, and
it needs an answer-quality harness that does not exist yet. Routing is what can
be scored honestly today, so routing is what is claimed.

## Consequences

**Every text question is rate-limited by Voyage.** `search_text` embeds the
query, and the unbilled free tier allows three requests a minute — the routing
eval hit the retry-and-backoff path repeatedly and took far longer than the
model calls warranted. The embedder's backoff absorbed it, but this is now an
interactive-latency problem rather than a batch one, and it is the strongest
argument yet for putting a payment method on the Voyage account: the 200M free
tokens still apply, so it removes the limit without adding cost.

Cost is roughly 5 cents a question, dominated by tool results in context.

`STRUCTURED_TOOLS` and `TEXT_TOOLS` classify every registered tool, and a test
asserts the classification is exhaustive. A tool added without being classified
would be invisible to the routing eval, which would go on reporting a score
while silently no longer covering the agent.

The tool runner requires tools to return **text**, not objects. Returning a dict
fails the request with a 400 that names neither the tool nor the cause; all six
tools serialise to JSON, and a test pins that so the failure cannot return
quietly.

This agent is not autonomous. It answers when asked and writes nothing. The
maintenance agent that opens pull requests unattended (PLAN §6) still needs
branch protection settled first.
