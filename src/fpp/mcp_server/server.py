"""MCP server exposing the match database as tools.

    python -m fpp.mcp_server        # stdio, which is how Claude Code connects

Deliberately thin. Every tool is a one-line call into tools.py, so the logic
worth testing is testable without an MCP client and the protocol layer adds
nothing that could hide a bug.

The docstring on each function below is what the model reads when choosing a
tool, so they are written for that reader: what it answers, when to reach for
it, and what it will not do. That is the part of this phase that is actually
hard -- see PLAN section 4 on tool ergonomics.
"""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from fpp.mcp_server import tools
from fpp.predict import MODEL_VERSION

server = MCPServer(
    name="football-predictor",
    version=MODEL_VERSION,
    instructions=(
        "Match outcome data and predictions for Europe's top five leagues, from "
        "1993 to the present. Every prediction is three probabilities -- home, "
        "draw, away -- which sum to 1; there is no single 'win probability'. "
        "Prefer the named tools over run_sql: they are faster, they resolve team "
        "aliases, and their output is shaped for reading. The database is opened "
        "read-only and nothing here can modify it. These are probabilistic "
        "estimates from historical results, not betting advice."
    ),
)


def _guard(fn, /, **kwargs) -> dict[str, Any]:
    """Turn a ToolError into a readable result instead of a stack trace.

    A caller that passed a bad team name needs the suggestion in the message,
    not a traceback it cannot act on.
    """
    try:
        return fn(**kwargs)
    except tools.ToolError as exc:
        return {"error": str(exc)}


@server.tool()
def predict_match(home: str, away: str) -> dict[str, Any]:
    """Predict any fixture: home win, draw and away win probabilities.

    Works for real fixtures and hypothetical ones. Returns expected goals and
    the likeliest scoreline as well as both teams' Elo ratings. Team names are
    resolved from common aliases ("Man Utd", "Spurs"); an unknown name comes
    back with the closest matches.

    For matches that are actually scheduled, prefer get_upcoming_fixtures --
    that returns the prediction recorded before kick-off, which is the one that
    counts. This tool always computes a fresh number.
    """
    return _guard(tools.predict_match, home=home, away=away)


@server.tool()
def get_upcoming_fixtures(division: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Scheduled fixtures with the predictions locked in before kick-off.

    division is a football-data.co.uk code -- E0 Premier League (the default),
    SP1 La Liga, I1 Serie A, D1 Bundesliga, F1 Ligue 1.

    An empty list is a normal answer, not a failure: the fixture source covers
    only about a week ahead, so there is genuinely nothing scheduled between
    rounds or before a season opens.
    """
    return _guard(tools.get_upcoming_fixtures, division=division, limit=limit)


@server.tool()
def get_head_to_head(team_a: str, team_b: str, limit: int = 10) -> dict[str, Any]:
    """Past meetings between two clubs, most recent first.

    Returns the win/draw/win record over the matches returned, each scoreline,
    and the de-vigged closing market probabilities where the source had them.
    Order of the two teams does not matter; both home and away meetings are
    included.

    Samples are small and mostly reflect squads that have since turned over, so
    this is context rather than a signal on its own.
    """
    return _guard(tools.get_head_to_head, team_a=team_a, team_b=team_b, limit=limit)


@server.tool()
def get_team_form(team: str, n: int = 10) -> dict[str, Any]:
    """Recent results for one club, plus its current Elo rating.

    Returns a W/D/L string with the most recent match first, goals scored and
    conceded over that span, and the individual matches.
    """
    return _guard(tools.get_team_form, team=team, n=n)


@server.tool()
def run_sql(query: str, limit: int = 100) -> dict[str, Any]:
    """Read-only SQL, for questions the four named tools do not cover.

    Reach for this last. The named tools resolve team aliases, shape their
    output, and cannot be pointed at a table scan; this cannot do any of that.
    Aggregations across many matches are the case it is genuinely for.

    SELECT and WITH only, one statement, at most 200 rows, five-second timeout.
    The connection is read-only, so writes fail in the driver regardless.

    Schema:
      teams(id, name, division)
      matches(id, division, season, match_date, home_team_id, away_team_id,
              home_goals, away_goals, result)      -- goals/result NULL if unplayed
      odds(match_id, source, odds_home, odds_draw, odds_away)  -- closing odds only
      predictions(id, match_id, model_version, prob_home, prob_draw, prob_away,
                  created_at)

    result is 'H', 'D' or 'A'. division is E0/SP1/I1/D1/F1. Team names live in
    teams.name; matches stores ids, so join through it.
    """
    return _guard(tools.run_sql, query=query, limit=limit)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
