-- Schema for the week 1-3 slice.
-- Deliberately plain SQL: no SQLite-only or Postgres-only syntax, so the
-- migration in ADR 0005 stays mechanical.

CREATE TABLE IF NOT EXISTS teams (
    id          INTEGER PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,  -- canonical name
    division    VARCHAR(10)                     -- most recent division seen
);

CREATE TABLE IF NOT EXISTS matches (
    id            INTEGER PRIMARY KEY,
    division      VARCHAR(10)  NOT NULL,
    season        VARCHAR(10)  NOT NULL,
    match_date    DATE         NOT NULL,
    home_team_id  INTEGER      NOT NULL REFERENCES teams(id),
    away_team_id  INTEGER      NOT NULL REFERENCES teams(id),
    home_goals    INTEGER,          -- NULL for a fixture not yet played
    away_goals    INTEGER,
    result        VARCHAR(1),       -- 'H' | 'D' | 'A', NULL if unplayed
    UNIQUE (division, season, match_date, home_team_id, away_team_id)
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches (match_date);
CREATE INDEX IF NOT EXISTS idx_matches_div_season ON matches (division, season);

-- Closing odds only. Opening odds exist in the source but the closing line is
-- the informed benchmark, so we do not store the opening one and invite its
-- accidental use. See CLAUDE.md on the B365H vs B365CH column trap.
CREATE TABLE IF NOT EXISTS odds (
    match_id    INTEGER PRIMARY KEY REFERENCES matches(id),
    source      VARCHAR(20) NOT NULL,   -- 'avg_closing' | 'b365_closing'
    odds_home   REAL,
    odds_draw   REAL,
    odds_away   REAL
);

-- Model output. Versioned so model generations can be compared against each
-- other and against what actually happened, without losing history.
CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY,
    match_id       INTEGER NOT NULL REFERENCES matches(id),
    model_version  VARCHAR(50) NOT NULL,
    prob_home      REAL NOT NULL,
    prob_draw      REAL NOT NULL,
    prob_away      REAL NOT NULL,
    created_at     TIMESTAMP NOT NULL,
    UNIQUE (match_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions (match_id);
