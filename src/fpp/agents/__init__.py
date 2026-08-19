"""Production agents. The first is entity resolution; the roster is in PLAN section 6."""

from fpp.agents.resolve_agent import (
    AGENT_VERSION,
    MODEL,
    AgentResolution,
    NameDecision,
    resolve_with_agent,
)

__all__ = [
    "AGENT_VERSION",
    "MODEL",
    "AgentResolution",
    "NameDecision",
    "resolve_with_agent",
]
