# 0011 — Purposeful MCP tools, and read-only enforced below the code

**Status:** accepted
**Date:** week 4

## Context

ADR 0003 split retrieval: structured facts are queried, unstructured text is retrieved. This builds the query side — an MCP server over the match database, exposing 17,939 matches, 161 teams and 12,459 closing prices to a model as tools.

`docs/PLAN.md` §10 wants it early specifically so it gets dogfooded: the same server is the development tool and the component the analyst agent will use in weeks 7–9. A tool layer designed without ever being used is designed blind.

## Decision

Four purposeful tools plus one deliberately inconvenient escape hatch — `predict_match`, `get_upcoming_fixtures`, `get_head_to_head`, `get_team_form`, and `run_sql`. The database is opened read-only at the connection. All logic lives in `mcp_server/tools.py` with no MCP imports; `server.py` is a thin adapter.

## Reasoning

**Named tools, because one `execute_sql` pushes the hard part onto the caller.** It requires schema knowledge to use, makes every call a possible table scan, returns shapes nobody designed, and leaves nothing to evaluate — PLAN §7 wants tool-call accuracy measured later, which presumes tools worth choosing between. The four here answer what this database actually gets asked. `run_sql` exists because fixed tools never cover everything, and its docstring tells the model to reach for it last.

**Read-only is enforced by the connection, not by the keyword filter — and this closes a gap in ADR 0009.** That ADR made a `PreToolUse` hook the structural guarantee against destructive operations. It inspects *shell commands*. An MCP tool call never touches a shell, so the guard cannot see it: the server is a write path the existing guardrail does not cover. So the engine is opened with SQLite's `mode=ro` URI, and a write fails inside `sqlite3` itself with "attempt to write a readonly database". The `SELECT`-only keyword check in `run_sql` runs first only to return a readable error instead of a driver exception; it is a courtesy, not the guarantee. `TestReadOnlyIsEnforcedBelowTheFilter` asserts the connection refuses writes directly, so a gap in the regex cannot pass as a green suite.

This distinction was worth getting right for a second reason. While building, the hook blocked an ad-hoc shell probe because it contained `DROP TABLE matches` as a *string* being passed to `run_sql` to prove it was rejected. The guard was behaving correctly by its own rules and the fix was to move the assertion into a test file, where it belonged anyway. A text-matching guard pushes destructive SQL out of shell history and into tests — a useful side effect, and a reminder of exactly how much that guard can be trusted to do.

**Lazy model fit, measured rather than assumed.** A Dixon-Coles fit over the pooled data takes 2.50s. Fitting per call would make the server unusable; fitting at import would make every session pay 2.5s so that one of five tools might respond faster. Three of the tools never need a model at all. So the fit is `lru_cache`d and happens on first use.

**`get_upcoming_fixtures` returns stored predictions, `predict_match` computes fresh ones.** For a scheduled match the locked row is what was forecast before kick-off (ADR 0010); recomputing it would report a number that was never the forecast. For a hypothetical pairing there is nothing stored and nothing to grade, so computing is correct. Each tool's docstring says which it does and points at the other.

**Team resolution lives in the error path.** Canonical names are not guessable — this database says `Tottenham`, not `Tottenham Hotspur`. Every tool resolves aliases first, then case, then near-misses, and an unknown name comes back as *"No team called 'Arsenl'. Closest matches: Arsenal."* An error the caller can act on is worth more than a separate discovery tool, and it keeps the tool count at the small number PLAN §4 asks for.

**SQLite, over `db.py`.** PLAN §8 says "MCP server over Postgres", but ADR 0005 settled on SQLite until pgvector is needed and nothing here needs it. Going through `db.py` keeps the server storage-agnostic, so the migration when RAG forces it is a change to one function.

## Alternatives considered

**A single `run_sql` tool and nothing else.** Rejected above. Kept as the escape hatch, capped at 200 rows with a five-second timeout enforced through SQLite's progress handler.

**A `list_teams` discovery tool.** Genuinely useful and rejected as padding. Folding the same information into failure messages means the caller only sees it when they need it, and does not spend a call to get it.

**Fitting models at server start.** Simpler code, worse behaviour: 2.5s added to every session for a cost three tools never incur.

**Migrating to Postgres first, to match PLAN §8 literally.** Rejected. The plan describes the end state; ADR 0005 governs when the move happens, and doing it now would be infrastructure ahead of a requirement.

## Consequences

`mcp>=2.0,<3` is a new runtime dependency, upper-bounded because the entry point moved between majors (`mcp.server.fastmcp` → `mcp.server.MCPServer`) — the same drift that motivated pinning ruff in ADR 0009.

`tools.py` importing no MCP symbols means the test suite exercises every tool without an MCP client, and the protocol layer stays thin enough that there is nothing in it worth testing.

`.mcp.json` names `.venv/Scripts/python.exe`, because the system interpreter has neither `mcp` nor `fpp` installed. That path is Windows-specific; a POSIX checkout needs `.venv/bin/python`, noted in the README.

There is still no agent. This is the tool layer only — the analyst that routes between these tools and RAG arrives in weeks 7–9, and building it now would leave it routing between one option and nothing.
