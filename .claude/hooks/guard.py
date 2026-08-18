#!/usr/bin/env python3
"""PreToolUse guard: no writes to main, no destructive SQL.

Wired to the Bash and PowerShell tools in .claude/settings.json. Reads the
hook payload on stdin and denies the call by printing a permissionDecision.

Why this exists (docs/PLAN.md section 6): every agent in the roadmap is
supposed to stop at a pull request and let a human merge. Enforcing that
through the agent's prompt does not hold — prompted-only restrictions fail
under pressure. This is the part that holds regardless of what the agent
concludes it should do.

What it is NOT: a security boundary. It matches command text, so anything
that builds a command dynamically or writes a script and runs it will get
through. It stops accidents and drift, which is what agents actually produce.
An adversary is out of scope, and pretending otherwise would be worse than
having no guard, because it would be trusted for something it cannot do.

Run `python .claude/hooks/guard.py --selftest` to check the rules still fire.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys

PROTECTED = {"main", "master"}

# Destructive SQL. DELETE and UPDATE are only flagged when unqualified, since a
# scoped DELETE is ordinary work and blocking it would make the guard something
# people route around.
SQL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bDROP\s+(TABLE|DATABASE|SCHEMA|INDEX|VIEW)\b", re.I), "DROP"),
    (re.compile(r"\bTRUNCATE\b", re.I), "TRUNCATE"),
    (re.compile(r"\bALTER\s+TABLE\b.*\bDROP\b", re.I | re.S), "ALTER TABLE ... DROP"),
    (re.compile(r"\bDELETE\s+FROM\b(?!.*\bWHERE\b)", re.I | re.S), "DELETE without WHERE"),
    # The lookbehind exempts `ON CONFLICT ... DO UPDATE SET`, which is an upsert
    # and has no WHERE by definition. This codebase uses that form in three
    # places; blocking it was a false positive, and a guard that blocks ordinary
    # work is a guard that gets switched off (ADR 0009).
    (
        re.compile(r"(?<!DO )\bUPDATE\b.+?\bSET\b(?!.*\bWHERE\b)", re.I | re.S),
        "UPDATE without WHERE",
    ),
]

# Commands that only read. Searching the codebase for the string "DROP TABLE"
# is not an attempt to drop a table, and a guard that blocks it teaches people
# to disable the guard.
READ_ONLY = re.compile(r"^\s*(grep|rg|ag|cat|head|tail|less|more|bat|git\s+grep|git\s+log)\b")


def deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def current_branch() -> str | None:
    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    try:
        out = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def segments(command: str) -> list[str]:
    """Split a shell line into individually-runnable pieces.

    `cd x && git push origin main` has to be caught, so the whole string
    cannot be treated as one command.
    """
    return [s.strip() for s in re.split(r"&&|\|\||;|\n|\|", command) if s.strip()]


def push_destinations(tokens: list[str]) -> list[str]:
    """Branches a `git push` would write to.

    Returns the destination side of each refspec, or the current branch when
    the push carries no refspec (`git push`, `git push origin`).
    """
    args = [t for t in tokens[2:] if not t.startswith("-")]
    refspecs = args[1:] if args else []
    if not refspecs:
        branch = current_branch()
        return [branch] if branch else []
    # origin main | origin HEAD:main | origin +feature:main
    return [r.split(":")[-1].lstrip("+") for r in refspecs]


def check_git(segment: str) -> None:
    tokens = segment.split()
    if len(tokens) < 2 or tokens[0] != "git":
        return
    verb = tokens[1]

    if verb == "push":
        for dest in push_destinations(tokens):
            if dest in PROTECTED:
                deny(
                    f"Blocked: this pushes to '{dest}'. Work reaches {dest} through a "
                    f"pull request that a human merges — see docs/PLAN.md section 6. "
                    f"Push a branch instead: git push -u origin <branch>"
                )

    if verb in {"merge", "rebase"}:
        branch = current_branch()
        if branch in PROTECTED:
            deny(
                f"Blocked: '{verb}' while on '{branch}' rewrites the protected branch. "
                f"Open a pull request and let a human merge it."
            )


def check_sql(segment: str) -> None:
    if READ_ONLY.match(segment):
        return
    # Collapse whitespace first: SQL in this project is written across several
    # lines, and the DO-UPDATE lookbehind is fixed-width so it only works
    # against a single space.
    normalized = " ".join(segment.split())
    for pattern, label in SQL_PATTERNS:
        if pattern.search(normalized):
            deny(
                f"Blocked: destructive SQL ({label}). The match database is rebuilt by "
                f"re-ingesting, not by editing in place — re-run scripts/ingest.py. If "
                f"this really is intended, run it yourself outside the agent."
            )


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # Never block on a payload we cannot parse.

    command = (payload.get("tool_input") or {}).get("command") or ""
    for segment in segments(command):
        check_git(segment)
        check_sql(segment)
    return 0


SELFTEST = [
    ("git push origin main", True),
    ("git push -u origin HEAD:main", True),
    ("git push --force origin +master", True),
    ("cd /repo && git push origin main", True),
    ("git push -u origin docs/my-branch", False),
    ("git push origin feature:feature", False),
    ('sqlite3 data/football.db "DROP TABLE matches"', True),
    ('sqlite3 data/football.db "DELETE FROM matches"', True),
    ('sqlite3 data/football.db "DELETE FROM matches WHERE id = 1"', False),
    ('sqlite3 data/football.db "SELECT count(*) FROM matches"', False),
    ('grep -rn "DROP TABLE" src/', False),
    ('sqlite3 db "UPDATE matches SET result = NULL"', True),
    ('sqlite3 db "UPDATE matches SET result = \'H\' WHERE id = 1"', False),
    # Upserts. Every ingester in this project ends with one of these, spread
    # over several lines; treating them as unqualified UPDATEs blocked ordinary
    # work until the DO-UPDATE exemption landed.
    ('python -c "ON CONFLICT (id) DO UPDATE SET result = NULL"', False),
    ('python -c "ON CONFLICT (id)\n  DO UPDATE SET\n  home_goals = excluded.home_goals"', False),
]


def selftest() -> int:
    failures = 0
    for command, should_block in SELFTEST:
        blocked = False
        try:
            # deny() writes the hook payload to stdout; swallow it so the
            # self-test reads as a result table rather than a JSON dump.
            with contextlib.redirect_stdout(io.StringIO()):
                for segment in segments(command):
                    check_git(segment)
                    check_sql(segment)
        except SystemExit:
            blocked = True
        status = "ok " if blocked == should_block else "FAIL"
        if blocked != should_block:
            failures += 1
        print(f"  {status}  {'block' if should_block else 'allow'}: {command}")
    print(f"\n{len(SELFTEST) - failures}/{len(SELFTEST)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    raise SystemExit(main())
