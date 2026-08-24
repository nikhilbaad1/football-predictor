"""Ingest football text from Wikipedia.

    https://en.wikipedia.org/w/api.php

Free, stable, and CC BY-SA — which carries attribution *and* share-alike, so the
licence is stored per document rather than assumed. Chosen over news RSS for the
first corpus because an encyclopedia article is still there next month: a
retrieval eval scored against a corpus that evaporates measures nothing twice.
News feeds arrive later, where PLAN section 11 limits us to headline, excerpt,
link and our own derived facts.

What goes in here is only what has no schema. Scorelines, dates and odds live in
`matches` and are queried exactly (ADR 0003); embedding them would convert a
lookup into an approximation. What Wikipedia adds is the prose around the
result — how a season went, what a club is, what happened and why.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date
from pathlib import Path
from typing import Any

import requests
from sqlalchemy import Engine, text

from fpp.config import RAW_DIR
from fpp.db import get_engine

log = logging.getLogger(__name__)

API = "https://en.wikipedia.org/w/api.php"
SOURCE = "wikipedia"
LICENSE = "CC BY-SA 4.0"
# Wikipedia asks for a descriptive agent identifying the client.
USER_AGENT = "football-predictor/0.1 (educational project; github.com/nikhilbaad1)"

# Articles change slowly and a retrieval eval wants a stable corpus, so pages are
# cached like the season CSVs: fetch once, reuse forever.
WIKI_CACHE = RAW_DIR / "wikipedia"
WIKI_CACHE.mkdir(parents=True, exist_ok=True)
REQUEST_DELAY = 0.5

# Sections that are navigation or citation apparatus, not content. Retrieving a
# reference list wastes the context window on things that answer nothing.
SKIP_HEADINGS = {
    "references", "external links", "see also", "notes", "further reading",
    "bibliography", "sources",
}

MIN_WORDS = 25      # below this a chunk is a stub heading, not a passage
MAX_WORDS = 220     # keeps a retrieved passage readable in a prompt

# Checking for "football club" alone is not enough, and this is not theoretical:
# it matched "Hull F.C.", a rugby league side, because English rugby league clubs
# are also named Football Club — putting 29 chunks of rugby into the corpus under
# a Premier League team. The sport has to be ruled out explicitly.
OTHER_SPORTS = ("rugby", "cricket", "basketball", "gaelic", "american football",
                "ice hockey", "baseball")

# Clubs whose canonical name here does not reach the right article by rule.
# Explicit rather than fuzzy, for the reason ingest/resolve.py refuses to guess:
# once a wrong article's text is in the corpus it retrieves well and reads
# plausibly, so nothing downstream ever surfaces the mistake.
PAGE_OVERRIDES = {
    "Hull": "Hull City A.F.C.",       # "Hull F.C." is the rugby league club
    "Ipswich": "Ipswich Town F.C.",   # no "Ipswich F.C." exists
}


def _cache_path(title: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", title)[:120]
    return WIKI_CACHE / f"{safe}.json"


def fetch_page(
    title: str,
    session: requests.Session | None = None,
    use_cache: bool = True,
    delay: float = REQUEST_DELAY,
    max_retries: int = 3,
) -> dict[str, Any] | None:
    """Fetch one article as plain text. Returns None if it does not exist.

    Cached to disk and rate-limited. Wikipedia answers 429 to a burst — which it
    did, the first time this ran across twenty clubs — and the fix is the same
    policy the CSV ingester already follows: fetch once, keep it, and space out
    what is genuinely new. Retries honour the server's Retry-After.
    """
    cache = _cache_path(title)
    if use_cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    session = session or requests.Session()
    resp = None
    for attempt in range(max_retries):
        time.sleep(delay)
        try:
            resp = session.get(
                API,
                params={
                    "action": "query", "prop": "extracts", "explaintext": 1,
                    "titles": title, "format": "json", "redirects": 1,
                },
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                log.warning("rate limited on %r; waiting %.1fs", title, wait)
                time.sleep(wait)
                resp = None
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            log.warning("could not fetch %r: %s", title, exc)
            return None
    if resp is None:
        log.warning("gave up on %r after %s attempts", title, max_retries)
        return None

    pages = resp.json().get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1" or "missing" in page:
            log.info("no such article: %r", title)
            return None
        extract = page.get("extract", "").strip()
        if not extract:
            return None
        result = {
            "title": page["title"],
            "text": extract,
            "url": f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}",
        }
        cache.write_text(json.dumps(result), encoding="utf-8")
        return result
    return None


def resolve_club_page(
    club: str, session: requests.Session | None = None
) -> dict[str, Any] | None:
    """Find the Wikipedia article for a club, or return None.

    A fourth naming vocabulary: this database says "Tottenham", Wikipedia says
    "Tottenham Hotspur F.C.", and a bare "Arsenal" is an article about weapons
    depots. Candidates are tried most-specific first and the result is checked
    to actually be about a football club, because a plausible wrong article is
    worse than none — it would fill the corpus with text that retrieves well and
    answers nothing.

    Same rule as ingest/resolve.py: no fuzzy fallback, no nearest title. An
    unresolved club is reported, not guessed at.
    """
    session = session or requests.Session()
    candidates = (
        [PAGE_OVERRIDES[club]] if club in PAGE_OVERRIDES
        else [f"{club} F.C.", f"{club} A.F.C.", club]
    )
    for candidate in candidates:
        page = fetch_page(candidate, session)
        if page and is_association_football_club(page["text"]):
            return page
    return None


def is_association_football_club(article: str) -> bool:
    """Is this article about a football club, and not another sport's club?

    Both halves are needed. The positive test alone admitted a rugby league
    side; the negative test alone would reject a football article that happens
    to mention a rugby ground.
    """
    opening = article[:800].lower()
    if any(sport in opening for sport in OTHER_SPORTS):
        return False
    return "football club" in opening


def _paragraphs(body: str) -> list[str]:
    """Paragraphs, with any single one longer than MAX_WORDS cut down to size.

    Packing between paragraphs is not enough on its own: one 600-word paragraph
    would sail past the cap and become a single chunk that fills a prompt and
    is mostly about something other than the query. Rare in practice, and the
    cap means nothing without it.
    """
    out: list[str] = []
    for para in (p.strip() for p in body.split("\n") if p.strip()):
        words = para.split()
        if len(words) <= MAX_WORDS:
            out.append(para)
            continue
        for i in range(0, len(words), MAX_WORDS):
            out.append(" ".join(words[i:i + MAX_WORDS]))
    return out


def chunk_text(raw: str) -> list[tuple[str | None, str]]:
    """Split an article into (heading, passage) pairs.

    Splits on Wikipedia's own section headings rather than a fixed window,
    because the headings are a real outline written by a person — passages that
    respect them stay about one thing, which is what makes a retrieved chunk
    answerable. Long sections are then split on paragraphs to stay readable.
    """
    parts = re.split(r"\n==+ *(.+?) *=+=\n", "\n" + raw)
    sections: list[tuple[str | None, str]] = [(None, parts[0])]
    for i in range(1, len(parts) - 1, 2):
        sections.append((parts[i].strip(), parts[i + 1]))

    chunks: list[tuple[str | None, str]] = []
    for heading, body in sections:
        if heading and heading.strip().lower() in SKIP_HEADINGS:
            continue
        buffer: list[str] = []
        count = 0
        for para in _paragraphs(body):
            words = len(para.split())
            if count and count + words > MAX_WORDS:
                chunks.append((heading, " ".join(buffer)))
                buffer, count = [], 0
            buffer.append(para)
            count += words
        if buffer:
            chunks.append((heading, " ".join(buffer)))

    return [(h, t) for h, t in chunks if len(t.split()) >= MIN_WORDS]


def store_document(
    page: dict[str, Any],
    engine: Engine | None = None,
    team: str | None = None,
    season: str | None = None,
    fetched_on: date | None = None,
) -> int:
    """Write one article and its chunks. Idempotent; re-ingest replaces chunks."""
    engine = engine or get_engine()
    chunks = chunk_text(page["text"])

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO documents (source, external_id, title, url, license,
                                       team, season, fetched_at)
                VALUES (:src, :eid, :title, :url, :lic, :team, :season, CURRENT_TIMESTAMP)
                ON CONFLICT (source, external_id) DO UPDATE SET
                    title = excluded.title, url = excluded.url,
                    team = excluded.team, season = excluded.season,
                    fetched_at = CURRENT_TIMESTAMP
                """
            ),
            {"src": SOURCE, "eid": page["title"], "title": page["title"],
             "url": page["url"], "lic": LICENSE, "team": team, "season": season},
        )
        doc_id = conn.execute(
            text("SELECT id FROM documents WHERE source = :s AND external_id = :e"),
            {"s": SOURCE, "e": page["title"]},
        ).scalar_one()

        # Replace rather than merge: an edited article can lose a section, and
        # leaving its chunks behind would keep serving text the source dropped.
        conn.execute(text("DELETE FROM chunks WHERE document_id = :d"), {"d": doc_id})
        for ordinal, (heading, body) in enumerate(chunks):
            conn.execute(
                text(
                    """
                    INSERT INTO chunks (document_id, ordinal, heading, text, n_words)
                    VALUES (:d, :o, :h, :t, :n)
                    """
                ),
                {"d": doc_id, "o": ordinal, "h": heading, "t": body,
                 "n": len(body.split())},
            )
    return len(chunks)


def ingest_pages(
    titles: dict[str, dict[str, Any]],
    engine: Engine | None = None,
) -> dict[str, int]:
    """Ingest several articles. `titles` maps page title -> {team, season}."""
    engine = engine or get_engine()
    session = requests.Session()
    counts: dict[str, int] = {}
    for title, meta in titles.items():
        page = fetch_page(title, session)
        if page is None:
            continue
        counts[title] = store_document(
            page, engine, team=meta.get("team"), season=meta.get("season")
        )
        log.info("%s: %s chunks", title, counts[title])
    return counts
