"""The entity-resolution agent: the project's first LLM call.

It answers exactly the question `ingest/resolve.py` cannot. That module resolves
what a rule can justify — exact match, alias table, club-type suffixes — and
refuses everything else, because deciding whether an unrecognised name is a new
club or a new spelling of a known one needs to know which football clubs exist.
This supplies that knowledge and nothing else.

Three things shape the design.

**It proposes; it does not write.** A wrong match merges two clubs' histories
and nothing downstream ever surfaces it. Decisions are returned for review and
only written when a caller explicitly applies them.

**Its output is validated, not trusted.** A `matched` decision naming a club
that is not in the known list is downgraded to `uncertain` rather than stored.
The model is a source of judgement, not a source of truth about our schema.

**"Uncertain" is a first-class answer.** An agent forced to choose will guess,
and a guess here is the exact failure the deterministic layer was built to
avoid. Escalating beats inventing.

Measured against the baseline in ADR 0012: 19 of 20 FPL names, zero wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from fpp.ingest.resolve import Resolution, resolve_team_name

log = logging.getLogger(__name__)

MODEL = "claude-opus-5"
AGENT_VERSION = "resolve-agent-v0.1"
MAX_TOKENS = 8000


class NameDecision(BaseModel):
    """The shape the model must answer in."""

    decision: Literal["matched", "new_entity", "uncertain"] = Field(
        description=(
            "matched: the same club as one in the known list, spelled differently. "
            "new_entity: a real club that is not in the known list. "
            "uncertain: cannot tell — escalate to a human."
        )
    )
    resolved_to: str | None = Field(
        default=None,
        description="Exact canonical name from the known list. Null unless decision is 'matched'.",
    )
    confidence: float = Field(description="0.0 to 1.0.")
    reasoning: str = Field(description="One or two sentences. What the club is, and why.")


@dataclass
class AgentResolution:
    """A proposed decision, plus what it cost to get."""

    raw: str
    decision: str
    resolved_to: str | None
    confidence: float
    reasoning: str
    model: str = MODEL
    agent_version: str = AGENT_VERSION
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    validation_note: str = ""

    @property
    def cost_usd(self) -> float:
        """Claude Opus 5 list pricing, cached tokens included.

        `usage.input_tokens` counts only uncached input, so adding just that to
        output would quietly understate every run — the roster is the bulk of
        the prompt and it is cached. Writes bill at 1.25x base, reads at 0.1x.
        """
        return (
            self.input_tokens * 5e-6
            + self.cache_write_tokens * 6.25e-6
            + self.cache_read_tokens * 0.5e-6
            + self.output_tokens * 25e-6
        )


SYSTEM = """You identify football clubs across Europe's top five leagues.

A database holds match history under a fixed set of canonical club names. Another
source has produced a name that does not match any of them by exact comparison,
by an alias table, or by ignoring club-type suffixes. Your job is to say which of
these it is:

  matched     — the same club as one already in the list, written differently
  new_entity  — a real club that genuinely is not in the list
  uncertain   — you cannot tell

What counts as the same club: abbreviations and nicknames; punctuation and
spacing differences; the presence or absence of a club-type word such as City,
Town, United, FC, or Albion; and ampersand versus "and".

What does NOT count as the same club: two different clubs based in the same town
or city, and two clubs whose names merely look similar. Sharing a place name is
not evidence of being the same club — many towns have more than one.

The two errors are not equally bad. Reporting a club as new when it already
exists creates a duplicate row that someone will notice and merge. Matching one
club to a different club silently fuses two teams' histories, and every number
computed afterwards is wrong with nothing to indicate it. When the evidence does
not clearly favour one existing club, answer new_entity or uncertain.

If you answer matched, resolved_to must be copied exactly from the known list."""


def _build_client(client=None):
    if client is not None:
        return client
    import anthropic  # imported lazily: nothing else in the project needs a key

    from fpp import config  # noqa: F401  — loads .env before the SDK reads the env

    return anthropic.Anthropic()


def resolve_with_agent(
    raw: str,
    known: set[str],
    client=None,
    model: str = MODEL,
) -> AgentResolution:
    """Ask the model to decide one unresolved name. Never writes anything."""
    deterministic: Resolution = resolve_team_name(raw, known)
    if deterministic.is_resolved:
        # Cheap and certain beats a model call. The agent exists for the residue.
        return AgentResolution(
            raw=raw, decision="matched", resolved_to=deterministic.resolved_to,
            confidence=deterministic.confidence,
            reasoning=f"Resolved without the model: {deterministic.rationale}",
            model="none", agent_version=AGENT_VERSION,
        )

    client = _build_client(client)
    roster = "\n".join(f"- {n}" for n in sorted(known))
    candidates = deterministic.candidates or ["(none close enough to list)"]

    response = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=[
            {"type": "text", "text": SYSTEM},
            # The roster is identical on every call and the name is not, so the
            # cache breakpoint goes here. Volatile content stays in the user turn.
            {"type": "text", "text": f"Known canonical club names:\n{roster}",
             "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{
            "role": "user",
            "content": (
                f"Name from the other source: {raw!r}\n\n"
                f"Nearest known names by string similarity, which may all be "
                f"irrelevant: {', '.join(candidates)}\n\n"
                f"Is this one of the known clubs, or a club not in the list?"
            ),
        }],
        output_format=NameDecision,
    )

    parsed: NameDecision = response.parsed_output
    result = AgentResolution(
        raw=raw,
        decision=parsed.decision,
        resolved_to=parsed.resolved_to,
        confidence=parsed.confidence,
        reasoning=parsed.reasoning,
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
    )
    return _validate(result, known)


def _validate(result: AgentResolution, known: set[str]) -> AgentResolution:
    """Check the decision against our own schema before anyone acts on it.

    The model supplies judgement about football, not facts about this database.
    A `matched` naming a club we do not have is a malformed answer, and treating
    it as a decision would write exactly the corruption this design prevents.
    """
    if result.decision == "matched":
        if not result.resolved_to:
            result.decision = "uncertain"
            result.validation_note = "matched without naming a club"
        elif result.resolved_to not in known:
            result.validation_note = (
                f"named {result.resolved_to!r}, which is not a known club"
            )
            result.decision = "uncertain"
            result.resolved_to = None
    elif result.resolved_to is not None:
        # new_entity/uncertain must not carry a target; drop it rather than let
        # a caller read it as a match.
        result.validation_note = f"dropped resolved_to on a {result.decision} decision"
        result.resolved_to = None
    return result
