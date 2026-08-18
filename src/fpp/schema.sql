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

-- Players, from the Fantasy Premier League API. Identity and availability only;
-- per-90 performance stats arrive with Understat/FBref in a later phase.
CREATE TABLE IF NOT EXISTS players (
    id           INTEGER PRIMARY KEY,
    fpl_id       INTEGER UNIQUE,          -- stable within a season, not across
    team_id      INTEGER REFERENCES teams(id),
    first_name   VARCHAR(100),
    second_name  VARCHAR(100),
    web_name     VARCHAR(100) NOT NULL,   -- what FPL displays
    position     VARCHAR(3)               -- GKP | DEF | MID | FWD
);

CREATE INDEX IF NOT EXISTS idx_players_team ON players (team_id);

-- Availability is a time series, not a column on players, and that is the whole
-- point. Expected minutes feeds prediction, so a backtest must be able to ask
-- what was known *before* a match rather than what turned out to be true. A
-- single mutable "is injured" flag would leak the future into every historical
-- fit -- see non-negotiable #1 in CLAUDE.md.
CREATE TABLE IF NOT EXISTS player_availability (
    player_id          INTEGER NOT NULL REFERENCES players(id),
    as_of              DATE    NOT NULL,
    status             VARCHAR(1),   -- a available, d doubtful, i injured, s suspended, u unavailable
    chance_next_round  INTEGER,      -- 0-100, NULL when the source says nothing
    news               TEXT,
    PRIMARY KEY (player_id, as_of)
);

CREATE INDEX IF NOT EXISTS idx_availability_asof ON player_availability (as_of);

-- Names a source used that the deterministic resolver would not commit to.
-- Refusing to guess is the design: a nearest-match rule that fixes "Hull City"
-- -> "Hull" will also merge "Coventry City" into "Leicester City", inventing a
-- link between two different clubs. Rows land here for judgement instead.
CREATE TABLE IF NOT EXISTS name_resolutions (
    id            INTEGER PRIMARY KEY,
    source        VARCHAR(20)  NOT NULL,  -- 'fpl' | 'understat' | ...
    raw_name      VARCHAR(100) NOT NULL,
    entity_type   VARCHAR(10)  NOT NULL,  -- 'team' | 'player'
    resolved_to   VARCHAR(100),           -- canonical name, NULL while unresolved
    decision      VARCHAR(12),            -- 'matched' | 'new_entity' | NULL if pending
    method        VARCHAR(20),            -- 'alias' | 'exact' | 'agent' | 'human'
    confidence    REAL,
    rationale     TEXT,
    created_at    TIMESTAMP NOT NULL,
    UNIQUE (source, raw_name, entity_type)
);
