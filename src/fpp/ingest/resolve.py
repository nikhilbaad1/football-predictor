"""Deterministic entity resolution across sources.

Every source spells clubs differently. football-data.co.uk says "Ath Madrid",
its own fixtures feed says "Atl. Madrid", FPL says "Spurs". Getting this wrong
does not raise -- it splits one club into two rows, and the model then rates a
team it has never seen and reports a confident, meaningless number.

The design rule here is **refuse to guess**.

A nearest-match rule tuned to fix "Hull City" -> "Hull" will also decide that
"Coventry City" is "Leicester City", because those two strings really are
similar. The first is a spelling difference; the second is a different football
club. No string metric separates them, because the difference is not in the
strings -- it is in knowing which clubs exist. So this module resolves only
what it can justify and hands everything else on as an open question, rather
than inventing a link that is very hard to notice afterwards.

This is deliberately the baseline the entity-resolution agent has to beat.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from fpp.ingest.teams import canonical_team_name

# Words that describe what kind of club it is rather than which club it is.
# "Hull City" and "Hull" are one club; the suffix carries no identity.
CLUB_SUFFIXES = {
    "fc", "afc", "cf", "sc", "ac", "united", "utd", "city", "town", "county",
    "albion", "wanderers", "rovers", "athletic", "hotspur", "rangers", "club",
}


@dataclass
class Resolution:
    """What the resolver concluded, and why.

    `decision` is "matched" or None. It is never "new_entity": deciding that a
    name is a club we have genuinely never seen requires knowing which clubs
    exist, which no rule in this module does. Absence of a match is reported as
    an open question, and something with world knowledge settles it.
    """

    raw: str
    resolved_to: str | None = None
    decision: str | None = None
    method: str = "unresolved"
    confidence: float = 0.0
    rationale: str = ""
    candidates: list[str] = field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        return self.decision == "matched"


def _head(name: str) -> str:
    """The identifying part of a club name, with type-words removed.

    "Hull City" -> "hull", "Ipswich Town" -> "ipswich". Note that this maps
    "Coventry City" to "coventry" and "Leicester City" to "leicester", which is
    exactly the point: the suffix is what those two share, and the head is what
    tells them apart.
    """
    words = [w for w in name.lower().replace(".", " ").split() if w]
    kept = [w for w in words if w not in CLUB_SUFFIXES]
    return " ".join(kept) if kept else " ".join(words)


def resolve_team_name(raw: str, known: set[str]) -> Resolution:
    """Resolve one source spelling against the canonical team names.

    Tried in order of how much each step can be justified: exact identity, the
    hand-kept alias table, then a head-word match that ignores club-type
    suffixes. Anything else is left open with its nearest candidates attached.
    """
    if not raw or not raw.strip():
        return Resolution(raw=raw, rationale="Empty name.")

    cleaned = " ".join(raw.split())

    if cleaned in known:
        return Resolution(cleaned, cleaned, "matched", "exact", 1.0,
                          "Identical to a known team name.")

    alias = canonical_team_name(cleaned)
    if alias in known:
        return Resolution(cleaned, alias, "matched", "alias", 1.0,
                          f"Alias table maps {cleaned!r} to {alias!r}.")

    # Head-word match. Requires a unique target: if "Manchester" were the head
    # of two known clubs, matching either would be a coin flip, so refuse.
    heads: dict[str, set[str]] = {}
    for name in known:
        heads.setdefault(_head(name), set()).add(name)

    target = heads.get(_head(alias), set())
    if len(target) == 1:
        match = next(iter(target))
        return Resolution(
            cleaned, match, "matched", "head_word", 0.9,
            f"{cleaned!r} and {match!r} share the identifying word "
            f"{_head(alias)!r}; the difference is a club-type suffix.",
        )
    ambiguous = len(target) > 1

    # Deliberately loose. These are suggestions for whoever decides next, never
    # a decision here, so recall matters and precision does not — an agent
    # reasoning "these are three different clubs, so this is a new one" needs to
    # see them. At a tighter cutoff "Coventry City" surfaces nothing at all,
    # which tells the next reader less than a list of near misses does.
    candidates = difflib.get_close_matches(alias, sorted(known), n=5, cutoff=0.5)
    return Resolution(
        raw=cleaned,
        method="unresolved",
        candidates=candidates,
        rationale=(
            f"{cleaned!r} matches {len(target)} known teams by head word, so a "
            f"choice between them would be arbitrary."
            if ambiguous else
            f"No exact, alias, or head-word match for {cleaned!r}. String "
            f"similarity alone cannot say whether this is a new club or a new "
            f"spelling of an existing one."
        ),
    )


def resolve_all(names: list[str], known: set[str]) -> dict[str, Resolution]:
    return {n: resolve_team_name(n, known) for n in names}
