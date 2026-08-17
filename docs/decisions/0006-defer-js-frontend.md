# 0006 — Server-rendered HTML now, defer the JS frontend

**Status:** accepted
**Date:** week 1

## Context

The architecture plan specifies Next.js. The week 1–3 deliverable is one page listing upcoming fixtures with three probabilities each.

## Decision

Serve that page from FastAPI with a Jinja2 template. Revisit Next.js in week 10 when there is a product to justify it.

## Reasoning

The week 1–3 page is a table. A Node toolchain, a second dependency tree, and a separate deployment target for a table is cost without benefit, and it lands in the phase where abandonment risk is highest.

Deferring also means the eventual frontend gets built against a real, stable API rather than one guessed at in week 1.

## Alternatives considered

**Next.js immediately.** Avoids rewriting the page later and gets deployment sorted early. The rewrite being discarded is roughly a hundred lines of template, so the saving is small. Rejected.

**No UI at all — CLI and JSON only.** Faster, but "predictions exist in my terminal" is meaningfully less demonstrable than a URL someone can open. Rejected.

## Consequences

The Jinja template is throwaway. That's acceptable at its size. The API is designed to be consumed by something else from the start — JSON endpoints exist alongside the HTML route rather than the page reading the database directly, so the later frontend has something to build against.
