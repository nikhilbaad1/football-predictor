# 0017 — Score the analyst on faithfulness, not accuracy

**Status:** accepted
**Date:** 2026-09-09

## Context

ADR 0016 shipped the analyst agent measuring only its *routing* — whether it went
to a source that could answer — and said plainly that answer correctness was not
measured. PLAN §7 wants Ragas in CI for exactly that gap.

There is a second reason this is worth doing carefully. ADR 0015 conceded that
the retrieval eval's relevance labels were written by the person who built the
retriever, after reading the corpus, with paraphrase cases deliberately
constructed to expose lexical retrieval's weakness. That is a real limit on what
those numbers mean. Adding another eval with the same flaw would compound it.

## Decision

Measure **faithfulness** — is every claim in the answer carried by the evidence
the agent itself retrieved? — rather than accuracy. Adopt Ragas's definition of
the metric; do not take the dependency.

## Reasoning

**Faithfulness needs no ground truth from me.** Accuracy requires someone to
decide the right answer, which is precisely the weakness ADR 0015 admitted.
Faithfulness compares an answer against the rows and passages the system itself
returned, so the judge needs no football knowledge and forms no opinion about
what is true — only whether the evidence entails the claim. The labels are not
mine, and the evidence is the system's own output.

**It tests the rule the whole retrieval layer rests on.** The analyst is
instructed never to answer from its own knowledge, and Claude knows a great deal
about football. If that instruction fails, the agent answers plausibly without
consulting anything and every ADR from 0011 onward describes decoration.
Faithfulness is how you find out. The judge is told explicitly that a claim it
knows to be true but which the evidence does not carry is *unsupported* — that
asymmetry is the metric.

**Ragas's definition, not Ragas.** Its faithfulness metric is exactly this:
decompose the answer into claims, check each against the retrieved context,
report the supported fraction. The definition is well-chosen and adopted. The
library is not, because Ragas models context as retrieved *text*, and half this
agent's evidence is structured JSON rows from tool calls. Forcing those into a
text-context shape would distort what is being judged, and the metric is forty
lines with structured output.

### The measurement

11 questions — 7 answerable from one or both sources, 4 whose facts exist in no
column and in no article.

```
faithfulness   99/103 claims supported (96%)
declined well  4/4 questions with no data answered without asserting anything unsupported
cost           $1.07
```

The four remaining unsupported claims are interpretive rather than fabricated —
the agent characterising the Arsenal–Chelsea rivalry in ways the passages
describe for *other* rivalries. That is a real finding: synthesis across
passages is where it drifts, not fact retrieval.

## Two mistakes in the first version, both mine

**The judge was given half the evidence.** The first run scored 84% and reported
four "invented" claims. Every one was a statement like *"the player data only
covers identity and availability"* or *"the match was a Premier League
fixture"* — the agent correctly declining and then explaining *why*. Those
statements are grounded in the **tool descriptions**, which state that E0 is the
Premier League and that `players` holds no minutes. The descriptions are part of
the agent's context; my evidence capture recorded only tool *results*. Including
them moved faithfulness from 84% to 96% and eliminated all four "inventions".

The lesson generalises: an eval that sees less than the system saw will report
the system as worse than it is, and the failures it invents look specific and
credible.

**The abstention metric measured the wrong thing.** It counted a question as
correctly declined only when the answer made *zero* claims. But an agent
answering *"there is no goalscorer data; here is what the database does hold"*
has declined perfectly — and scored 0/4. The metric now asks whether the answer
asserted anything the evidence cannot carry, which is what actually matters, and
the same set scores 4/4.

I nearly shipped "the agent invents things on questions it cannot answer". It
does not. Both numbers were artefacts of the harness.

## Alternatives considered

**Score accuracy against reference answers.** The measure everyone wants, and it
reintroduces exactly the ground-truth problem ADR 0015 documented. Worth doing
later with labels from someone other than the system's author.

**Take the Ragas dependency.** Standard, comparable to published numbers, and a
poor fit for structured tool results — plus a large dependency in a project
whose thesis is that its components can be explained under questioning.

**Judge with a cheaper model.** The judge decides whether text entails a claim,
which is the capability being bought. At a dollar per full run, saving on it is
not worth the risk of a metric that quietly gets easier.

## Consequences

Faithful is not correct. A claim the source got wrong is still supported, and the
script says so on every run. Source accuracy is a separate question this does not
touch.

The judge is strict about inference. It flagged *"Saka is listed as belonging to
Arsenal"* when the query returned `team_id: 1` and never joined it to a name.
That is technically right and arguably pedantic, and it is the correct direction
to err in for a metric whose purpose is catching ungrounded claims.

Evidence is captured through a `ContextVar` rather than a module global, so
concurrent `ask()` calls cannot read each other's evidence, and a tool called
outside an `ask()` records nothing.

Evidence entries are labelled with the **tool** name passed explicitly, not
`fn.__name__`. The underlying function is not always named what the agent called,
and the log is read by a judge and by people.

This still runs only when invoked. Putting it in CI needs an API key in GitHub
Actions secrets, which is the same blocker as the maintenance agent.
