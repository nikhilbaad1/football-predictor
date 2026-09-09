"""Faithfulness judge tests.

No API calls. What the judge concludes about real answers is measured by
`scripts/eval_answers.py`, which costs money; what is tested here is the
scoring, the evidence assembly, and the one omission that skewed the first run.
"""

from __future__ import annotations

import json

import pytest

from fpp.agents.analyst import tool_descriptions
from fpp.agents.judge import (
    MAX_EVIDENCE_CHARS,
    ClaimCheck,
    Faithfulness,
    FaithfulnessReport,
    format_evidence,
    judge_faithfulness,
)


def check(verdict: str, claim: str = "c") -> ClaimCheck:
    return ClaimCheck(claim=claim, verdict=verdict, evidence_quote="q")


class TestScoring:
    def test_all_supported_scores_one(self):
        f = Faithfulness("q", [check("supported"), check("supported")])
        assert f.score == 1.0
        assert f.unfaithful == []

    def test_partial_support(self):
        f = Faithfulness("q", [check("supported"), check("unsupported")])
        assert f.score == pytest.approx(0.5)
        assert len(f.unfaithful) == 1

    def test_contradicted_counts_against(self):
        f = Faithfulness("q", [check("supported"), check("contradicted")])
        assert f.score == pytest.approx(0.5)

    def test_an_answer_with_no_claims_scores_one(self):
        """"The tools do not have this" is the right answer to an unanswerable
        question and asserts nothing to be unfaithful about. Whether it *should*
        have declined is a separate question the eval asks, not this score."""
        assert Faithfulness("q", []).score == 1.0

    def test_cost_is_reported(self):
        f = Faithfulness("q", [], input_tokens=1000, output_tokens=100)
        assert f.cost_usd == pytest.approx(1000 * 5e-6 + 100 * 25e-6)


class TestEvidenceAssembly:
    def test_tool_results_are_labelled(self):
        out = format_evidence([{"tool": "get_team_form", "result": '{"form":"WWD"}'}])
        assert "get_team_form" in out
        assert "WWD" in out

    def test_tool_descriptions_are_included_when_given(self):
        """The omission that skewed the first run.

        The tool docstrings tell the agent that E0 is the Premier League and
        that `players` holds no minutes. Judging without them marked an agent
        explaining its own limits as one inventing claims — faithfulness went
        from 96% to 84% purely on this.
        """
        out = format_evidence([], tool_docs="E0 is the Premier League")
        assert "E0 is the Premier League" in out

    def test_docs_survive_alongside_results(self):
        out = format_evidence(
            [{"tool": "run_sql", "result": "{}"}], tool_docs="SCHEMA HERE"
        )
        assert "SCHEMA HERE" in out and "run_sql" in out

    def test_no_evidence_is_stated_not_faked(self):
        assert "no tools" in format_evidence([])

    def test_dict_results_are_serialised(self):
        out = format_evidence([{"tool": "t", "result": {"a": 1}}])
        assert '"a": 1' in out

    def test_oversized_evidence_says_it_was_truncated(self):
        """A silently trimmed context makes supported claims look unsupported,
        which would depress the score for a reason nobody could see."""
        out = format_evidence([{"tool": "t", "result": "x" * (MAX_EVIDENCE_CHARS + 100)}])
        assert "[EVIDENCE TRUNCATED]" in out


class TestToolDescriptions:
    def test_reads_the_real_docstrings(self):
        """@beta_tool wraps the function, so the docstring lives on
        `.description`. Reading `__doc__` yields the wrapper's and silently
        produces a near-empty context."""
        docs = tool_descriptions()
        assert len(docs) > 1500
        assert docs.count("--- tool:") == 6

    def test_carries_the_facts_the_judge_needs(self):
        docs = tool_descriptions()
        assert "E0 Premier League" in docs, "division codes must be grounded"
        assert "players(id, fpl_id" in docs, "schema must be grounded"


class FakeParsed:
    def __init__(self, claims):
        self.parsed_output = FaithfulnessReport(claims=claims)
        self.usage = type("U", (), {"input_tokens": 10, "output_tokens": 5})()


class FakeClient:
    def __init__(self, claims):
        self._claims = claims
        self.kwargs = None
        self.messages = self

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return FakeParsed(self._claims)


class TestJudge:
    def test_returns_the_claims_and_usage(self):
        client = FakeClient([check("supported"), check("unsupported")])
        out = judge_faithfulness("q", "an answer", [], client=client)
        assert out.total == 2
        assert out.supported == 1
        assert out.input_tokens == 10

    def test_an_empty_answer_costs_nothing(self):
        client = FakeClient([check("supported")])
        out = judge_faithfulness("q", "   ", [], client=client)
        assert out.claims == []
        assert client.kwargs is None, "called the model for an empty answer"

    def test_the_prompt_forbids_using_outside_knowledge(self):
        """The rule the whole metric rests on: a claim the judge happens to know
        is true, but which the evidence does not carry, must be unsupported."""
        client = FakeClient([])
        judge_faithfulness("q", "a", [], client=client)
        system = client.kwargs["system"]
        assert "Do not use your own knowledge" in system
        assert "true in the real world" in system

    def test_evidence_and_answer_both_reach_the_model(self):
        client = FakeClient([])
        judge_faithfulness("why?", "because X", [{"tool": "t", "result": "R"}],
                           client=client, tool_docs="DOCS")
        content = client.kwargs["messages"][0]["content"]
        assert "because X" in content and "R" in content and "DOCS" in content

    def test_structured_output_is_requested(self):
        client = FakeClient([])
        judge_faithfulness("q", "a", [], client=client)
        assert client.kwargs["output_format"] is FaithfulnessReport


class TestEvidenceCaptureInTheAgent:
    def test_ask_records_what_each_tool_returned(self, monkeypatch):
        """Faithfulness can only be judged against what the agent actually saw,
        and the tool runner does not hand results back to the caller."""
        from fpp.agents import analyst

        monkeypatch.setattr(analyst.db, "get_team_form", lambda **kw: {"form": "WWD"})
        collected: list[dict] = []
        token = analyst._EVIDENCE.set(collected)
        try:
            analyst.get_team_form("Arsenal")
        finally:
            analyst._EVIDENCE.reset(token)

        assert len(collected) == 1
        assert collected[0]["tool"] == "get_team_form"
        assert json.loads(collected[0]["result"])["form"] == "WWD"

    def test_recording_is_off_outside_an_ask(self, monkeypatch):
        """No ambient global: a tool called on its own records nothing."""
        from fpp.agents import analyst

        monkeypatch.setattr(analyst.db, "get_team_form", lambda **kw: {"form": "W"})
        assert analyst._EVIDENCE.get() is None
        analyst.get_team_form("Arsenal")  # must not raise
