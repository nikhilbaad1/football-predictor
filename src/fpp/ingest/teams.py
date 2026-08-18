"""Team-name normalization.

Sources spell the same club differently ("Man United", "Manchester Utd",
"Man Utd"). This module owns the canonical name. Right now it is a hand-kept
alias table, which is fine for five leagues and one source.

This is deliberately the *seed* of the entity-resolution problem that the
ingestion agent takes over in week 4-6. Keeping it explicit and testable now
means there is something concrete for that agent to improve on, and a baseline
to measure it against.
"""

from __future__ import annotations

ALIASES: dict[str, str] = {
    # England
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "manchester utd": "Manchester United",
    "man city": "Manchester City",
    "newcastle": "Newcastle United",
    "nott'm forest": "Nottingham Forest",
    "notts forest": "Nottingham Forest",
    "sheffield weds": "Sheffield Wednesday",
    "sheffield utd": "Sheffield United",
    "wolves": "Wolverhampton Wanderers",
    "west brom": "West Bromwich Albion",
    "west ham": "West Ham United",
    "spurs": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "leeds": "Leeds United",
    "leicester": "Leicester City",
    "norwich": "Norwich City",
    "brighton": "Brighton & Hove Albion",
    "qpr": "Queens Park Rangers",
    # Spain
    # The fixtures feed and the season files disagree here: results say
    # "Ath Madrid", fixtures say "Atl. Madrid". Missing this split the club in
    # two and produced a confident prediction for a team with no history.
    "ath madrid": "Atletico Madrid",
    "atl. madrid": "Atletico Madrid",
    "atl madrid": "Atletico Madrid",
    "atletico madrid": "Atletico Madrid",
    "ath bilbao": "Athletic Bilbao",
    "espanol": "Espanyol",
    "sociedad": "Real Sociedad",
    "betis": "Real Betis",
    "vallecano": "Rayo Vallecano",
    "celta": "Celta Vigo",
    "la coruna": "Deportivo La Coruna",
    # Italy
    "inter": "Inter Milan",
    "internazionale": "Inter Milan",
    "milan": "AC Milan",
    "roma": "AS Roma",
    "verona": "Hellas Verona",
    # Germany
    "bayern munich": "Bayern Munich",
    "ein frankfurt": "Eintracht Frankfurt",
    "leverkusen": "Bayer Leverkusen",
    "dortmund": "Borussia Dortmund",
    "m'gladbach": "Borussia Monchengladbach",
    "mgladbach": "Borussia Monchengladbach",
    "hertha": "Hertha Berlin",
    "fc koln": "FC Koln",
    "hoffenheim": "TSG Hoffenheim",
    # France
    "paris sg": "Paris Saint-Germain",
    "psg": "Paris Saint-Germain",
    "st etienne": "Saint-Etienne",
    "marseille": "Olympique Marseille",
    "lyon": "Olympique Lyonnais",
}


def canonical_team_name(raw: str) -> str:
    """Map a source spelling to the canonical club name.

    Unknown names pass through cleaned rather than raising — a new promoted club
    should not break ingestion. Review `SELECT name FROM teams ORDER BY name`
    after ingesting a new season to catch spellings that need an alias.
    """
    if raw is None:
        raise ValueError("team name is None")
    cleaned = " ".join(str(raw).split()).strip()
    return ALIASES.get(cleaned.lower(), cleaned)
