"""The analyst agent: routes a question to structured data or to text.

This is ADR 0003 in action, and the reason that ADR was worth writing. A
question like "how many times have Arsenal beaten Chelsea" has an exact answer
in `matches` and should be counted, not retrieved — approximate nearest
neighbour is strictly worse at producing it. A question like "why did the club
move stadium" has no schema at all and can only be retrieved. The agent's job is
to know which is which.

It reuses `mcp_server.tools` as plain Python rather than speaking MCP. That is
the payoff for the rule in CLAUDE.md that the tool logic imports nothing from
`mcp`: the same five tools serve an MCP client and an in-process agent, and this
module needs no server, no subprocess and no protocol to be tested.

Two things it will not do. It does not write — every tool here reads, and the
database connection underneath is opened read-only (ADR 0011). And it does not
answer from its own knowledge of football: the system prompt requires that every
claim come from a tool result, because a model that already knows who won the
2023-24 title will happily answer without checking, and then the retrieval layer
this project spent two ADRs measuring is decoration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from anthropic import beta_tool

from fpp.mcp_server import tools as db
from fpp.rag.hybrid import hybrid_search

log = logging.getLogger(__name__)

MODEL = "claude-opus-5"
AGENT_VERSION = "analyst-v0.1"
MAX_TOKENS = 8000
MAX_ITERATIONS = 8

# Which retrieval path each tool represents. The routing eval scores against
# this, so it lives with the tools rather than in the eval script.
STRUCTURED_TOOLS = {
    "predict_match", "get_upcoming_fixtures", "get_head_to_head",
    "get_team_form", "run_sql",
}
TEXT_TOOLS = {"search_text"}


def _safe(fn, **kwargs) -> str:
    """Run a tool and hand back JSON text.

    The tool runner requires a string, not a dict. A caller-fixable problem is
    returned as a result rather than raised: the agent can read "closest
    matches: Arsenal" and try again, which a traceback does not allow.
    """
    try:
        payload = fn(**kwargs)
    except db.ToolError as exc:
        payload = {"error": str(exc)}
    return json.dumps(payload, default=str)


@beta_tool
def predict_match(home: str, away: str) -> str:
    """Predict a fixture: home win, draw and away win probabilities.

    Computed from Elo and Dixon-Coles over historical results. Also returns
    expected goals, the likeliest scoreline, and both teams' Elo ratings.
    Team aliases like "Man Utd" and "Spurs" are resolved automatically.

    Use for any question about who is likely to win a match, real or
    hypothetical. Not for questions about matches already played.
    """
    return _safe(db.predict_match, home=home, away=away)


@beta_tool
def get_upcoming_fixtures(division: str | None = None, limit: int = 20) -> str:
    """Scheduled fixtures with the predictions recorded before kick-off.

    division is a football-data.co.uk code: E0 Premier League (default), SP1
    La Liga, I1 Serie A, D1 Bundesliga, F1 Ligue 1.

    An empty list is a normal answer — the fixture source publishes only about a
    week ahead, so there is genuinely nothing scheduled between rounds.
    """
    return _safe(db.get_upcoming_fixtures, division=division, limit=limit)


@beta_tool
def get_head_to_head(team_a: str, team_b: str, limit: int = 10) -> str:
    """Past meetings between two clubs, most recent first.

    Returns the win/draw/win record, every scoreline, and the de-vigged closing
    market probabilities where the source had them. Order of the teams does not
    matter.

    Use for any question about results between two specific clubs — how many
    times one has beaten the other, recent scorelines, historical record.
    """
    return _safe(db.get_head_to_head, team_a=team_a, team_b=team_b, limit=limit)


@beta_tool
def get_team_form(team: str, n: int = 10) -> str:
    """Recent results for one club, plus its current Elo rating.

    Returns a W/D/L string with the most recent match first, goals scored and
    conceded over that span, and the individual matches.
    """
    return _safe(db.get_team_form, team=team, n=n)


@beta_tool
def run_sql(query: str, limit: int = 100) -> str:
    """Read-only SQL over the match database, for counts and aggregations the
    other tools do not cover.

    Reach for this only when a named tool cannot answer the question. SELECT and
    WITH only, one statement, at most 200 rows, five-second timeout.

    Schema:
      teams(id, name, division)
      matches(id, division, season, match_date, home_team_id, away_team_id,
              home_goals, away_goals, result)   -- goals/result NULL if unplayed
      odds(match_id, source, odds_home, odds_draw, odds_away)
      predictions(id, match_id, model_version, prob_home, prob_draw, prob_away,
                  created_at)
      players(id, fpl_id, team_id, first_name, second_name, web_name, position)
      player_availability(player_id, as_of, status, chance_next_round, news)

    result is 'H', 'D' or 'A'. Team names live in teams.name; matches stores ids.
    player_availability is a dated time series — filter by as_of.
    """
    return _safe(db.run_sql, query=query, limit=limit)


@beta_tool
def search_text(query: str, team: str | None = None, limit: int = 5) -> str:
    """Search Wikipedia club and season articles for descriptive text.

    Use for anything with no exact answer in the database: club history,
    stadiums, nicknames, ownership, rivalries, why something happened, what a
    season was like. Pass `team` (a canonical club name) to restrict the search
    when the question is about one club.

    This contains prose only. It does NOT contain scorelines, dates, league
    tables or odds — those are in the database, and asking here for them returns
    text that mentions numbers rather than the numbers themselves.
    """
    hits = hybrid_search(query, team=team, limit=limit)
    return json.dumps({
        "count": len(hits),
        "passages": [
            {
                "title": h["title"], "heading": h["heading"],
                "text": h["text"], "url": h["url"],
            }
            for h in hits
        ],
        "license": "Text from Wikipedia, CC BY-SA 4.0.",
    }, default=str)


ALL_TOOLS = [
    predict_match, get_upcoming_fixtures, get_head_to_head,
    get_team_form, run_sql, search_text,
]

SYSTEM = """You answer questions about football using the tools provided, and
only using the tools provided.

Two kinds of question, and choosing correctly matters:

  Exact facts — results, counts, scorelines, dates, league positions, odds,
  probabilities, player availability — live in a database. Use the structured
  tools. Never estimate a number that can be counted.

  Descriptive text — history, stadiums, nicknames, ownership, rivalries, why
  something happened — has no schema and lives in article text. Use search_text.

A question may need both, and then you should use both.

You almost certainly already know a great deal about football. Do not use it.
Every factual claim in your answer must come from a tool result in this
conversation. If the tools do not answer the question, say so plainly and say
what you tried — that is a useful answer, and inventing one is not.

Say where each fact came from: name the club article for retrieved text, and say
"from the match database" for structured results. Probabilities are estimates
from historical results, not predictions of what will happen, and never betting
advice. Keep answers short."""


@dataclass
class AnalystAnswer:
    question: str
    answer: str
    tools_used: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: str = MODEL
    agent_version: str = AGENT_VERSION

    @property
    def used_structured(self) -> bool:
        return bool(set(self.tools_used) & STRUCTURED_TOOLS)

    @property
    def used_text(self) -> bool:
        return bool(set(self.tools_used) & TEXT_TOOLS)

    @property
    def route(self) -> str:
        """Which retrieval path the agent actually took."""
        if self.used_structured and self.used_text:
            return "both"
        if self.used_structured:
            return "structured"
        if self.used_text:
            return "text"
        return "none"

    @property
    def cost_usd(self) -> float:
        """Claude Opus 5 list pricing, cached tokens included."""
        return (
            self.input_tokens * 5e-6
            + self.cache_write_tokens * 6.25e-6
            + self.cache_read_tokens * 0.5e-6
            + self.output_tokens * 25e-6
        )


def _build_client(client=None):
    if client is not None:
        return client
    import anthropic

    from fpp import config  # noqa: F401  — loads .env before the SDK reads it

    return anthropic.Anthropic()


def ask(question: str, client=None, model: str = MODEL) -> AnalystAnswer:
    """Answer one question, routing between structured tools and text search."""
    client = _build_client(client)
    result = AnalystAnswer(question=question, answer="", model=model)

    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        tools=ALL_TOOLS,
        messages=[{"role": "user", "content": question}],
    )

    final_text: list[str] = []
    for iteration, message in enumerate(runner):
        usage = getattr(message, "usage", None)
        if usage:
            result.input_tokens += getattr(usage, "input_tokens", 0) or 0
            result.output_tokens += getattr(usage, "output_tokens", 0) or 0
            result.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            result.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

        for block in message.content:
            if block.type == "tool_use":
                result.tools_used.append(block.name)
            elif block.type == "text" and block.text.strip():
                final_text = [block.text.strip()]

        if iteration >= MAX_ITERATIONS:
            # A loop that will not settle is a bug worth surfacing, not one to
            # let run up a bill.
            log.warning("stopping after %s iterations", MAX_ITERATIONS)
            break

    result.answer = final_text[0] if final_text else ""
    return result
