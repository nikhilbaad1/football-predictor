"""FastAPI app: JSON endpoints plus one server-rendered page.

The HTML page reads from the JSON layer rather than the database directly, so
the eventual JS frontend (ADR 0006) has a real API to build against instead of
one invented later.

Models are fitted once at startup and held in memory. At this data size a fit
takes seconds and the parameters change only when new results land, so fitting
per request would be wasteful. A scheduled refit replaces this in week 4+.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from fpp.config import DISPLAY_DIVISION, DIVISIONS
from fpp.db import load_matches
from fpp.predict import MODEL_VERSION, Fixture, fit_models, predict_fixtures

log = logging.getLogger(__name__)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

STATE: dict = {"elo": None, "dc": None, "teams": [], "fitted_at": None, "n_train": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        played = load_matches(divisions=list(DIVISIONS), played_only=True)
        if played.empty:
            log.warning("database is empty — run scripts/ingest.py")
        else:
            elo, dc = fit_models(played)
            display_teams = load_matches(divisions=[DISPLAY_DIVISION], played_only=True)
            recent_season = display_teams["season"].max()
            recent = display_teams[display_teams["season"] == recent_season]
            STATE.update(
                elo=elo,
                dc=dc,
                teams=sorted(set(recent["home_team"]) | set(recent["away_team"])),
                fitted_at=date.today(),
                n_train=len(played),
            )
            log.info("fitted on %s matches, %s display teams", len(played), len(STATE["teams"]))
    except Exception as exc:
        log.exception("startup fit failed: %s", exc)
    yield


app = FastAPI(title="Football Predictor", version=MODEL_VERSION, lifespan=lifespan)


def _require_models():
    if STATE["elo"] is None or STATE["dc"] is None:
        raise HTTPException(
            status_code=503,
            detail="Models not fitted. Run scripts/ingest.py, then restart.",
        )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if STATE["elo"] is not None else "no_models",
        "model_version": MODEL_VERSION,
        "training_matches": STATE["n_train"],
        "fitted_at": str(STATE["fitted_at"]) if STATE["fitted_at"] else None,
    }


@app.get("/api/teams")
def teams() -> dict:
    _require_models()
    return {"division": DIVISIONS[DISPLAY_DIVISION], "teams": STATE["teams"]}


@app.get("/api/predict")
def predict(
    home: str = Query(..., description="Home team, canonical name"),
    away: str = Query(..., description="Away team, canonical name"),
) -> dict:
    """Three-way probabilities for one fixture."""
    _require_models()
    if home == away:
        raise HTTPException(status_code=400, detail="A team cannot play itself")

    df = predict_fixtures([Fixture(home, away)], STATE["elo"], STATE["dc"])
    row = df.iloc[0].to_dict()
    row.pop("match_date", None)
    row["note"] = (
        "Probabilistic estimate from historical results. Not betting advice."
    )
    return row


@app.get("/api/ratings")
def ratings(limit: int = 30) -> dict:
    """Current Elo ratings for the displayed division."""
    _require_models()
    elo = STATE["elo"]
    rated = sorted(
        ((t, round(elo.rating(t), 1)) for t in STATE["teams"]),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return {"model_version": MODEL_VERSION, "ratings": [
        {"rank": i + 1, "team": t, "elo": r} for i, (t, r) in enumerate(rated[:limit])
    ]}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """The week 1-3 deliverable: one page, real probabilities."""
    fixtures: list[dict] = []
    if STATE["elo"] is not None and len(STATE["teams"]) >= 2:
        teams = STATE["teams"]
        elo = STATE["elo"]
        ranked = sorted(teams, key=elo.rating, reverse=True)
        # No forward-fixture feed yet, so demonstrate on notable pairings among
        # the strongest sides. Replaced by real fixtures when a source lands.
        top = ranked[:8]
        pairs = [(top[i], top[i + 1]) for i in range(0, min(len(top) - 1, 7))]
        fixtures = predict_fixtures(
            [Fixture(h, a) for h, a in pairs], STATE["elo"], STATE["dc"]
        ).to_dict("records")

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "fixtures": fixtures,
            "division": DIVISIONS[DISPLAY_DIVISION],
            "model_version": MODEL_VERSION,
            "n_train": STATE["n_train"],
        },
    )
