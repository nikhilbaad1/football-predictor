"""Analyst agent tests.

No API calls. Routing quality is measured by `scripts/eval_analyst.py`, which
costs money and is run deliberately; what is tested here is everything around
the model call.

The invariant worth naming is TestEveryToolIsClassified. The routing eval scores
against STRUCTURED_TOOLS and TEXT_TOOLS, so a tool added without being put in one
of them would be silently invisible to the measurement — the eval would keep
reporting a score while no longer covering the agent.
"""

from __future__ import annotations

import json

import pytest

from fpp.agents import analyst
from fpp.agents.analyst import (
    ALL_TOOLS,
    STRUCTURED_TOOLS,
    TEXT_TOOLS,
    AnalystAnswer,
    ask,
)


def tool_names() -> set[str]:
    return {t.name if hasattr(t, "name") else t.__name__ for t in ALL_TOOLS}


class TestEveryToolIsClassified:
    def test_no_tool_is_left_unrouted(self):
        """Otherwise the routing eval scores a smaller agent than the one that
        ships, and keeps reporting a number while doing it."""
        assert tool_names() == STRUCTURED_TOOLS | TEXT_TOOLS

    def test_the_two_classes_do_not_overlap(self):
        assert not (STRUCTURED_TOOLS & TEXT_TOOLS)


class TestToolsReturnText:
    """The tool runner requires a string; returning a dict fails the request
    with a 400 that names neither the tool nor the cause."""

    def test_a_structured_tool_returns_json_text(self, monkeypatch):
        monkeypatch.setattr(analyst.db, "get_team_form", lambda **kw: {"team": "Arsenal"})
        out = analyst.get_team_form("Arsenal")
        assert isinstance(out, str)
        assert json.loads(out) == {"team": "Arsenal"}

    def test_the_text_tool_returns_json_text(self, monkeypatch):
        monkeypatch.setattr(analyst, "hybrid_search", lambda *a, **kw: [
            {"title": "Arsenal F.C.", "heading": "Stadiums", "text": "…",
             "url": "https://example.invalid"}
        ])
        payload = json.loads(analyst.search_text("stadium"))
        assert payload["count"] == 1
        assert "CC BY-SA" in payload["license"], "attribution must survive the tool boundary"

    def test_a_tool_error_becomes_a_result_not_an_exception(self, monkeypatch):
        """The agent can read 'closest matches: Arsenal' and retry. It cannot
        read a traceback."""
        def boom(**kw):
            raise analyst.db.ToolError("No team called 'Arsenl'. Closest matches: Arsenal.")

        monkeypatch.setattr(analyst.db, "predict_match", boom)
        payload = json.loads(analyst.predict_match("Arsenl", "Chelsea"))
        assert "Arsenal" in payload["error"]

    def test_non_serialisable_values_do_not_break_the_tool(self, monkeypatch):
        """Dates come back from the database as date objects."""
        from datetime import date
        monkeypatch.setattr(analyst.db, "get_head_to_head",
                            lambda **kw: {"when": date(2026, 5, 1)})
        assert json.loads(analyst.get_head_to_head("a", "b"))["when"] == "2026-05-01"


class TestRoute:
    @pytest.mark.parametrize(
        "tools,expected",
        [
            (["get_head_to_head"], "structured"),
            (["run_sql", "get_team_form"], "structured"),
            (["search_text"], "text"),
            (["get_head_to_head", "search_text"], "both"),
            ([], "none"),
        ],
    )
    def test_route_reflects_the_tools_used(self, tools, expected):
        assert AnalystAnswer("q", "a", tools_used=tools).route == expected

    def test_cost_counts_cached_tokens(self):
        """input_tokens excludes cached input; ignoring the rest understates
        every run, and the system prompt is the same on every call."""
        a = AnalystAnswer("q", "a", input_tokens=100, output_tokens=50,
                          cache_write_tokens=200, cache_read_tokens=800)
        expected = 100 * 5e-6 + 200 * 6.25e-6 + 800 * 0.5e-6 + 50 * 25e-6
        assert a.cost_usd == pytest.approx(expected)


class Block:
    def __init__(self, type_, name=None, text=None):
        self.type = type_
        self.name = name
        self.text = text


class Message:
    def __init__(self, blocks, usage=None):
        self.content = blocks
        self.usage = usage


class Usage:
    def __init__(self, i=10, o=5):
        self.input_tokens = i
        self.output_tokens = o
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0


class FakeRunner:
    def __init__(self, messages):
        self._messages = messages

    def __iter__(self):
        return iter(self._messages)


class FakeClient:
    def __init__(self, messages):
        self._messages = messages
        self.kwargs = None
        self.beta = self
        self.messages = self

    def tool_runner(self, **kwargs):
        self.kwargs = kwargs
        return FakeRunner(self._messages)


class TestAsk:
    def test_records_the_tools_and_the_final_text(self):
        client = FakeClient([
            Message([Block("tool_use", name="get_head_to_head")], Usage()),
            Message([Block("text", text="Arsenal have won 10.")], Usage()),
        ])
        out = ask("how many?", client=client)
        assert out.tools_used == ["get_head_to_head"]
        assert out.answer == "Arsenal have won 10."
        assert out.route == "structured"

    def test_the_last_text_block_wins(self):
        """Intermediate commentary before a tool call is not the answer."""
        client = FakeClient([
            Message([Block("text", text="Let me look that up.")], Usage()),
            Message([Block("tool_use", name="search_text")], Usage()),
            Message([Block("text", text="They left for capacity reasons.")], Usage()),
        ])
        assert ask("why?", client=client).answer == "They left for capacity reasons."

    def test_tokens_accumulate_across_turns(self):
        client = FakeClient([Message([Block("text", text="x")], Usage(10, 5))] * 3)
        out = ask("q", client=client)
        assert out.input_tokens == 30
        assert out.output_tokens == 15

    def test_all_tools_and_the_system_prompt_are_sent(self):
        client = FakeClient([Message([Block("text", text="x")], Usage())])
        ask("q", client=client)
        assert len(client.kwargs["tools"]) == len(ALL_TOOLS)
        assert "only using the tools provided" in client.kwargs["system"]

    def test_a_runaway_loop_is_stopped(self):
        """A loop that will not settle is a bug to surface, not a bill to run up."""
        client = FakeClient([Message([Block("tool_use", name="run_sql")], Usage())] * 50)
        out = ask("q", client=client)
        assert len(out.tools_used) <= analyst.MAX_ITERATIONS + 1

    def test_no_text_block_yields_an_empty_answer(self):
        client = FakeClient([Message([Block("tool_use", name="run_sql")], Usage())])
        assert ask("q", client=client).answer == ""
