# 0001 — Record architecture decisions

**Status:** accepted
**Date:** week 1

## Context

This project is built with heavy AI assistance and is intended for a CV and technical interviews. The value of it as a portfolio piece rests almost entirely on demonstrable judgment — the decisions about what to build, what to reject, and why. Code alone doesn't show that, because AI-assisted code looks the same whether the human made the calls or not.

## Decision

Keep a numbered decision log in `docs/decisions/`. Write an entry whenever a choice has a real alternative. Record the alternative and the reasoning, not just the outcome.

## Alternatives considered

**Rely on commit messages.** Too granular and too focused on *what* changed rather than *why* one approach was chosen over another. Rejected.

**Write it up retroactively before interviews.** This is the failure mode the log exists to prevent — reconstructed reasoning is thin and unconvincing, and by then the discarded alternatives have been forgotten. Rejected.

## Consequences

A small ongoing cost per decision. In exchange, "why did you do it this way?" always has a specific answer with the rejected options attached.
