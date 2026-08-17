# 0002 — Premier League first, La Liga second

**Status:** accepted
**Date:** week 1

## Context

The project began with Real Madrid vs Barcelona as the motivating example, implying La Liga as the pilot league. On a free-data budget, the available player-level data differs sharply by league.

## Decision

Build the Premier League first. Add La Liga once the pipeline works.

## Reasoning

The Fantasy Premier League API (`fantasy.premierleague.com/api/bootstrap-static/`) is official, free, requires no key, requires no scraping, and carries per-player injury status and a "chance of playing next round" field.

That last field matters more than it appears. Player goal probability is modelled as a Poisson process where expected minutes is a direct multiplier on λ — a 20-minute substitute has roughly a fifth of a starter's exposure. Without an availability feed, expected minutes has to be guessed from recent appearances, which is exactly the kind of hand-wave that makes a model look rigorous while being arbitrary underneath.

No free source covers La Liga equivalently.

## Alternatives considered

**La Liga first, guessing expected minutes from recent starts.** Preserves the original example at the cost of a weak input to the player model. Rejected — the player layer is a stated goal, not a bonus.

**Both leagues at once.** Doubles ingestion and team-name normalization work in week 1, before either is validated. Rejected.

## Consequences

The Clásico example doesn't work until La Liga lands. Model *training* still pools the top-5 leagues (see 0004), so La Liga data is ingested for training well before it's displayed.
