"""Ingest players and availability from the Fantasy Premier League API.

    https://fantasy.premierleague.com/api/bootstrap-static/

Official, free, no key, no scraping. One request returns every team and player
with, crucially, availability: `status`, `chance_of_playing_next_round` and a
`news` string. Nothing free covers that for any other league, which is most of
why ADR 0002 put the Premier League first.

Two things this module will not do.

**It does not guess team identity.** Names arrive in a third vocabulary -- FPL
says "Spurs" and "Man Utd" -- and anything the deterministic resolver will not
commit to is written to name_resolutions and its players are skipped. A wrong
match here silently merges two clubs' squads.

**It does not overwrite availability history.** Each run writes a snapshot
stamped with today's date. Expected minutes feeds prediction, so a backtest has
to be able to ask what was known *before* a match rather than what turned out
to be true; a single mutable "injured" flag would leak the future into every
historical fit.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import requests
from sqlalchemy import Engine, text

from fpp.db import get_engine
from fpp.ingest.resolve import Resolution, resolve_team_name

log = logging.getLogger(__name__)

FPL_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
SOURCE = "fpl"


def fetch_bootstrap(url: str = FPL_URL) -> dict[str, Any] | None:
    """Fetch the FPL snapshot. Not cached -- availability is the point of it."""
    log.info("fetching %s", url)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("could not fetch %s: %s", url, exc)
        return None
    return resp.json()


def known_team_names(conn) -> set[str]:
    return {r[0] for r in conn.execute(text("SELECT name FROM teams"))}


def _record_resolution(conn, res: Resolution, entity_type: str = "team") -> None:
    """Log what the resolver concluded, including what it declined to decide.

    Unresolved rows are the agent's queue. Resolved ones are kept too, so the
    agent can later be scored against decisions already known to be right.
    """
    conn.execute(
        text(
            """
            INSERT INTO name_resolutions
                (source, raw_name, entity_type, resolved_to, decision, method,
                 confidence, rationale, created_at)
            VALUES (:src, :raw, :etype, :to, :dec, :method, :conf, :why,
                    CURRENT_TIMESTAMP)
            ON CONFLICT (source, raw_name, entity_type) DO UPDATE SET
                resolved_to = excluded.resolved_to,
                decision    = excluded.decision,
                method      = excluded.method,
                confidence  = excluded.confidence,
                rationale   = excluded.rationale
            """
        ),
        {
            "src": SOURCE, "raw": res.raw, "etype": entity_type,
            "to": res.resolved_to, "dec": res.decision, "method": res.method,
            "conf": res.confidence, "why": res.rationale,
        },
    )


def ingest_fpl(
    engine: Engine | None = None,
    data: dict[str, Any] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Load FPL teams, players and availability. Idempotent.

    Returns counts plus the list of team names left unresolved, which the caller
    is expected to surface rather than swallow.
    """
    engine = engine or get_engine()
    today = today or date.today()
    if data is None:
        data = fetch_bootstrap()
    if not data or "teams" not in data or "elements" not in data:
        return {"players": 0, "availability": 0, "unresolved": []}

    positions = {
        et["id"]: et.get("singular_name_short", "")
        for et in data.get("element_types", [])
    }

    players = availability = 0
    unresolved: list[str] = []

    with engine.begin() as conn:
        known = known_team_names(conn)

        # Resolve every FPL team first. Players are only written for teams whose
        # identity is settled, so an unresolved club costs its squad rather than
        # attaching it to the wrong team.
        fpl_team_to_id: dict[int, int] = {}
        for team in data["teams"]:
            res = resolve_team_name(team["name"], known)
            _record_resolution(conn, res)
            if not res.is_resolved:
                unresolved.append(team["name"])
                log.warning("unresolved FPL team %r: %s", team["name"], res.rationale)
                continue
            tid = conn.execute(
                text("SELECT id FROM teams WHERE name = :n"), {"n": res.resolved_to}
            ).scalar_one()
            fpl_team_to_id[team["id"]] = tid

        for el in data["elements"]:
            tid = fpl_team_to_id.get(el.get("team"))
            if tid is None:
                continue  # squad of an unresolved club

            conn.execute(
                text(
                    """
                    INSERT INTO players (fpl_id, team_id, first_name, second_name,
                                         web_name, position)
                    VALUES (:fid, :tid, :first, :second, :web, :pos)
                    ON CONFLICT (fpl_id) DO UPDATE SET
                        team_id     = excluded.team_id,
                        first_name  = excluded.first_name,
                        second_name = excluded.second_name,
                        web_name    = excluded.web_name,
                        position    = excluded.position
                    """
                ),
                {
                    "fid": el["id"], "tid": tid,
                    "first": el.get("first_name"), "second": el.get("second_name"),
                    "web": el.get("web_name") or "unknown",
                    "pos": positions.get(el.get("element_type"), ""),
                },
            )
            pid = conn.execute(
                text("SELECT id FROM players WHERE fpl_id = :f"), {"f": el["id"]}
            ).scalar_one()
            players += 1

            chance = el.get("chance_of_playing_next_round")
            conn.execute(
                text(
                    """
                    INSERT INTO player_availability
                        (player_id, as_of, status, chance_next_round, news)
                    VALUES (:p, :d, :s, :c, :n)
                    ON CONFLICT (player_id, as_of) DO UPDATE SET
                        status            = excluded.status,
                        chance_next_round = excluded.chance_next_round,
                        news              = excluded.news
                    """
                ),
                {
                    "p": pid, "d": today,
                    "s": (el.get("status") or "")[:1] or None,
                    "c": int(chance) if chance is not None else None,
                    "n": (el.get("news") or "").strip() or None,
                },
            )
            availability += 1

    return {"players": players, "availability": availability, "unresolved": unresolved}


def pending_resolutions(engine: Engine | None = None) -> list[dict[str, Any]]:
    """Names no rule would commit to. This is the agent's inbox."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT source, raw_name, entity_type, method, rationale
                FROM name_resolutions
                WHERE decision IS NULL
                ORDER BY source, raw_name
                """
            )
        ).mappings().all()
    return [dict(r) for r in rows]
