"""Faithfulness judging: is every claim in an answer supported by the evidence?

ADR 0016 shipped the analyst agent scoring only its *routing* — whether it went
to a source that could answer — and said plainly that answer correctness was not
measured. This measures the part of correctness that can be measured honestly.

**Why faithfulness and not accuracy.** Scoring accuracy needs someone to decide
the right answer, and ADR 0015 already conceded how weak that ground truth is
when the person writing the labels is the person who built the system.
Faithfulness asks a different question with no such dependency: *does every
claim in this answer follow from the passages and rows the system itself
retrieved?* The evidence is the system's own output, so the judge needs no
knowledge of football and no opinion about what is true — only whether the text
entails the claim. An answer can be faithful and wrong (if the source is wrong),
and that is a property of the source, not of the agent.

It also tests the one prompt rule that the whole retrieval layer rests on. The
analyst is told never to answer from its own knowledge of football, and Claude
knows a great deal about football. Faithfulness is how you find out whether that
instruction held.

**Following Ragas's definition, without the dependency.** PLAN section 7 names
Ragas, and its faithfulness metric is exactly this: decompose the answer into
claims, check each against the retrieved context, score the supported fraction.
The definition is adopted; the library is not, because Ragas models context as
retrieved *text* and half this agent's evidence is structured JSON rows from
tool calls, which does not fit that shape without distorting it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

MODEL = "claude-opus-5"
JUDGE_VERSION = "faithfulness-v0.1"
MAX_TOKENS = 8000
MAX_EVIDENCE_CHARS = 60_000


class ClaimCheck(BaseModel):
    claim: str = Field(description="One factual claim, quoted or closely paraphrased.")
    verdict: Literal["supported", "unsupported", "contradicted"] = Field(
        description=(
            "supported: the evidence states or directly entails it. "
            "unsupported: the evidence neither states nor entails it — including "
            "claims that are true in the world but absent from the evidence. "
            "contradicted: the evidence says otherwise."
        )
    )
    evidence_quote: str = Field(
        description="Short quote from the evidence, or why nothing supports it."
    )


class FaithfulnessReport(BaseModel):
    claims: list[ClaimCheck] = Field(
        description="Every factual claim in the answer. Empty if it makes none."
    )


@dataclass
class Faithfulness:
    question: str
    claims: list[ClaimCheck]
    input_tokens: int = 0
    output_tokens: int = 0
    judge_version: str = JUDGE_VERSION

    @property
    def total(self) -> int:
        return len(self.claims)

    @property
    def supported(self) -> int:
        return sum(1 for c in self.claims if c.verdict == "supported")

    @property
    def unfaithful(self) -> list[ClaimCheck]:
        return [c for c in self.claims if c.verdict != "supported"]

    @property
    def score(self) -> float:
        """Supported fraction. An answer making no claims scores 1.0.

        That is deliberate: "the tools do not have this" is the correct answer
        to an unanswerable question, and it asserts nothing to be unfaithful
        about. Whether the agent *should* have abstained is a separate check.
        """
        return 1.0 if not self.claims else self.supported / self.total

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * 5e-6 + self.output_tokens * 25e-6


SYSTEM = """You check whether an answer is supported by the evidence it was
built from. You are not checking whether the answer is true.

You will be given EVIDENCE — the raw results of database queries and text
passages that a system retrieved — and an ANSWER written from it.

Break the answer into its individual factual claims. For each one, decide
whether the EVIDENCE supports it:

  supported     — the evidence states it, or it follows directly from the
                  evidence (including arithmetic over rows that are present)
  unsupported   — the evidence neither states nor entails it. Use this even when
                  the claim is true in the real world. A claim you happen to
                  know is correct, but which is absent from the evidence, is
                  unsupported — that is the entire point of this check.
  contradicted  — the evidence says otherwise

Do not use your own knowledge of the subject to justify a claim. The only
question is whether the evidence in front of you carries it.

Ignore things that are not factual claims: hedges, offers to help, restatements
of the question, and explicit statements that the data is unavailable. Attribution
like "from the match database" is not a claim. If the answer makes no factual
claims at all, return an empty list."""


def _build_client(client=None):
    if client is not None:
        return client
    import anthropic

    from fpp import config  # noqa: F401  — loads .env before the SDK reads it

    return anthropic.Anthropic()


def format_evidence(evidence: list[dict], tool_docs: str | None = None) -> str:
    """Flatten the evidence the agent saw into text the judge can read.

    `tool_docs` matters more than it looks. The tool descriptions are part of
    the agent's context and state things like "E0 is the Premier League" and
    what `players` does not hold. Omitting them made the judge mark an agent
    explaining its own limits as one inventing claims.
    """
    header = (
        f"--- what the tools are and what they hold ---\n{tool_docs}\n\n"
        if tool_docs else ""
    )
    if not evidence:
        return header + "(no tools returned any result)"
    parts = []
    for i, item in enumerate(evidence, start=1):
        result = item.get("result", "")
        if isinstance(result, (dict, list)):
            result = json.dumps(result, default=str)
        parts.append(f"--- evidence {i}: {item.get('tool', 'unknown')} ---\n{result}")
    joined = header + "\n\n".join(parts)
    if len(joined) > MAX_EVIDENCE_CHARS:
        # Truncating evidence makes supported claims look unsupported, so say so
        # rather than let a silently trimmed context depress the score.
        joined = joined[:MAX_EVIDENCE_CHARS] + "\n\n[EVIDENCE TRUNCATED]"
        log.warning("evidence truncated at %s chars", MAX_EVIDENCE_CHARS)
    return joined


def judge_faithfulness(
    question: str,
    answer: str,
    evidence: list[dict],
    client=None,
    model: str = MODEL,
    tool_docs: str | None = None,
) -> Faithfulness:
    """Check every claim in `answer` against the evidence the agent retrieved."""
    if not answer.strip():
        return Faithfulness(question=question, claims=[])

    client = _build_client(client)
    response = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"QUESTION\n{question}\n\n"
                f"EVIDENCE\n{format_evidence(evidence, tool_docs)}\n\n"
                f"ANSWER\n{answer}"
            ),
        }],
        output_format=FaithfulnessReport,
    )
    return Faithfulness(
        question=question,
        claims=response.parsed_output.claims,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
