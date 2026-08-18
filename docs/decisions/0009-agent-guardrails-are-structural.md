# 0009 — Agent guardrails are structural, not prompted

**Status:** accepted
**Date:** week 3 (agents arrive weeks 4+)

## Context

The repository is now on GitHub, and the roadmap's next phase introduces agents that write code and open pull requests. `docs/PLAN.md` section 6 sets the autonomy model: every agent that touches code or promotes a model stops at a pull request, and merging is a human decision.

That model needs two things that did not exist yet — a way to tell mechanically whether a pull request is safe to merge, and an enforcement point that does not depend on an agent's cooperation.

## Decision

Build both before the first agent, not alongside it.

1. **CI on every pull request** (`.github/workflows/ci.yml`): lint, tests on Python 3.10 and 3.13, and the no-lookahead leakage check as its own separately-named job.
2. **A `PreToolUse` hook** (`.claude/hooks/guard.py`, wired in `.claude/settings.json`) that denies pushes and merges to `main` and denies destructive SQL, for the interactive session and every agent that runs in this repository.
3. **An explicit ruff rule set** in `pyproject.toml` (`E`, `F`, `I`, `B`, `UP`) with the version pinned to `>=0.16,<0.17`.

## Reasoning

**Prompted restrictions do not hold, and the failure is silent.** An instruction in a system prompt is advice the model weighs against everything else in its context. Under pressure — a failing check, an ambiguous instruction, a long session — it loses. The hook is a separate process that reads the command and returns a denial; it does not care what the agent concluded. This is the same reason the leakage invariant in `evaluation/backtest.py` is enforced by structure and a test rather than by a comment saying not to leak.

**The leakage check gets its own CI job even though `tests` already runs it.** Redundant by execution, not by purpose: a required status check named `no-lookahead` states the invariant on the pull request page, where a reviewer sees it. Buried inside "74 passed" it communicates nothing, and this is the one failure where the correct response is to stop and disbelieve every number the project reports.

**The ruff rule set is pinned because a lint gate that changes on its own is worse than no gate.** The declared dependency was `ruff>=0.3` with no `select`, so the effective rule set was whatever the installed ruff defaulted to — and those defaults have expanded across releases. Running 0.16 against code written against an older default produced 24 findings that were never opted into. A gate agents depend on cannot move underneath them, so the rules are now the project's explicit choice and the version has an upper bound.

**Python 3.10 and 3.13 in the matrix**, because 3.10 is the floor `pyproject.toml` declares and 3.13 is what development actually happens on. Testing only the development version makes the declared floor a fiction.

## Alternatives considered

**Rely on GitHub branch protection alone.** Necessary but not sufficient, and it is the wrong layer on its own. Branch protection stops the push at the server after the agent has already decided to make it; the hook stops the decision from becoming an action, locally, with an error message that tells the agent what to do instead. They also fail differently — protection is bypassable by a repository admin, the hook is bypassable by dynamic command construction. Both, not either. Branch protection is still worth enabling and is listed under Consequences.

**Have the hook block a wider set of git operations** — `reset --hard`, `branch -D`, force-push on any branch. Rejected as scope. Force-pushing a feature branch is ordinary work, and a guard that blocks ordinary work gets disabled, at which point it protects nothing. The rule is limited to what section 6 actually specifies: pushes and merges to a protected branch, plus destructive SQL.

**Block all `DELETE` and `UPDATE`.** Rejected for the same reason — a scoped `DELETE ... WHERE` is normal. Only the unqualified forms are denied, along with `DROP`, `TRUNCATE`, and `ALTER TABLE ... DROP`.

## Consequences

**The hook is not a security boundary, and is documented in its own source as not being one.** It matches command text, so anything that builds a command dynamically, writes a script and runs it, or renames a binary gets through. It stops accidents and drift, which is what agents actually produce. Claiming more would be worse than having no guard, because it would then be trusted for something it cannot do.

**Branch protection on `main` is still required and is not yet enabled** — it needs repository settings access, which is a human action. Until it is on, the hook is the only thing standing between an agent and `main`.

**Hook changes need a Claude Code restart to take effect** when `.claude/` did not exist at session start; the settings watcher only tracks directories that had settings when the session began.

Fixing the lint findings changed three `zip()` calls in `models/blend.py` and `models/dixon_coles.py` to pass `strict=True`. Those are provably no-ops today — each zips sequences whose lengths are checked or derived from the same `n` — but they convert a future parameter-layout mistake from silently dropping teams into a loud failure, which is the behaviour the testing policy in `CLAUDE.md` asks for.
