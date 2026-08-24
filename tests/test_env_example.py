"""Guard against a real secret reaching a committed file.

This exists because it nearly happened. An Anthropic key was pasted into
`.env.example` instead of `.env` — a tracked file, one `git add -A` away from
being on GitHub permanently. It was caught before the commit by luck rather than
by anything in the repository, which is not a control.

Rotating a leaked key is easy; noticing it leaked is the hard part, and a public
repo means the window between push and scrape is short. So the check runs in CI
on every push.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Vendor-prefixed key shapes. A placeholder ends in "..." and never matches,
# because the run of key characters after the prefix is too short.
SECRET_PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{24,}"), "Anthropic API key"),
    (re.compile(r"\bpa-[A-Za-z0-9_\-]{24,}"), "Voyage API key"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "GitHub token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
]

SKIP_DIRS = {".git", ".venv", "venv", "data", "__pycache__", ".pytest_cache",
             ".ruff_cache", "node_modules", "htmlcov"}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".sql",
                 ".html", ".txt", ".cfg", ".ini", ".example", ".sh"}
MAX_BYTES = 2_000_000


def _scannable_files() -> list[Path]:
    out = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name == ".env":       # gitignored by design; not committed
            continue
        if path.suffix in TEXT_SUFFIXES or path.name.startswith(".env"):
            if path.stat().st_size <= MAX_BYTES:
                out.append(path)
    return out


class TestNoSecretsInCommittedFiles:
    def test_env_example_has_no_real_key(self):
        """The exact mistake that happened, asserted directly."""
        content = (ROOT / ".env.example").read_text(encoding="utf-8")
        for pattern, label in SECRET_PATTERNS:
            assert not pattern.search(content), (
                f"{label} found in .env.example. Real keys belong in .env, "
                f"which is gitignored. Move it and rotate the key."
            )

    def test_placeholders_are_obviously_placeholders(self):
        """Every key line in the template should be unusable as-is, so a reader
        cannot mistake a stale example for a working value."""
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            name, _, value = stripped.partition("=")
            if not name.strip().endswith(("_KEY", "_TOKEN", "_SECRET")):
                continue
            assert value.strip().endswith("..."), (
                f"{name.strip()} in .env.example should be a '...' placeholder, "
                f"got something that looks like a value"
            )

    def test_no_secret_anywhere_in_the_working_tree(self):
        """Broader net: a key pasted into any source or doc file, not just the
        template. Skips .env itself, which is gitignored by design."""
        offenders = []
        for path in _scannable_files():
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern, label in SECRET_PATTERNS:
                if pattern.search(content):
                    offenders.append(f"{label} in {path.relative_to(ROOT)}")
        assert not offenders, "secrets found in tracked files: " + "; ".join(offenders)


class TestEnvIsIgnored:
    def test_gitignore_covers_dotenv(self):
        """If .env stops being ignored, every key in it is one commit from
        GitHub and nothing else in the repo would notice."""
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert ".env" in {line.strip() for line in ignored}

    def test_env_example_is_not_ignored(self):
        """The template is meant to be committed — that is the whole point of
        keeping real values out of it."""
        ignored = {line.strip() for line in
                   (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()}
        assert ".env.example" not in ignored


class TestPatternsActuallyMatch:
    """A guard that never fires is indistinguishable from a broken one."""

    @pytest.mark.parametrize(
        "sample",
        [
            "sk-ant-api03-" + "A" * 40,
            "pa-" + "b" * 30,
            "ghp_" + "C" * 36,
            "AKIA" + "D" * 16,
        ],
    )
    def test_a_realistic_key_is_detected(self, sample):
        assert any(p.search(sample) for p, _ in SECRET_PATTERNS)

    @pytest.mark.parametrize("sample", ["sk-ant-...", "pa-...", "ghp_...", "not-a-key"])
    def test_placeholders_are_not_flagged(self, sample):
        assert not any(p.search(sample) for p, _ in SECRET_PATTERNS)
