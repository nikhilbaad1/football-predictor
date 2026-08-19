"""Entity-resolution agent tests.

No API calls: CI has no key, and a test whose result depends on a model call
measures the model rather than the code. Scoring the model is a separate job
with a separate tool — `scripts/eval_resolver.py`, which costs money and is run
deliberately.

What matters here is everything around the call: that a rule-resolvable name
never reaches the model, that a malformed answer is caught before it reaches the
database, and that a settled decision is never silently overwritten.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from fpp.agents.resolve_agent import (
    AgentResolution,
    NameDecision,
    _validate,
    resolve_with_agent,
)
from fpp.db import init_db
from fpp.ingest.fpl import apply_resolution, ingest_fpl, pending_resolutions

KNOWN = {"Arsenal", "Chelsea", "Tottenham", "Sheffield United", "Hull"}


class FakeUsage:
    def __init__(self):
        self.input_tokens = 100
        self.output_tokens = 50
        self.cache_creation_input_tokens = 200
        self.cache_read_input_tokens = 800


class FakeResponse:
    def __init__(self, decision):
        self.parsed_output = decision
        self.usage = FakeUsage()


class FakeClient:
    """Records that it was called, and with what."""

    def __init__(self, decision: NameDecision):
        self._decision = decision
        self.calls = 0
        self.last_kwargs = None
        self.messages = self

    def parse(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return FakeResponse(self._decision)


def decision(dec, target=None, conf=0.9):
    return NameDecision(
        decision=dec, resolved_to=target, confidence=conf, reasoning="because"
    )


class TestTheModelIsOnlyAskedWhenNeeded:
    def test_a_rule_resolvable_name_never_reaches_the_model(self):
        """Cheap and certain beats a model call, and it beats paying for one."""
        client = FakeClient(decision("new_entity"))
        out = resolve_with_agent("Spurs", KNOWN, client=client)
        assert client.calls == 0
        assert out.resolved_to == "Tottenham"
        assert out.model == "none"
        assert out.cost_usd == 0.0

    def test_an_unresolvable_name_does_reach_the_model(self):
        client = FakeClient(decision("new_entity"))
        resolve_with_agent("Coventry City", KNOWN, client=client)
        assert client.calls == 1

    def test_the_roster_is_sent_and_cached(self):
        """The club list is identical on every call and the name is not, so the
        cache breakpoint belongs on the roster."""
        client = FakeClient(decision("new_entity"))
        resolve_with_agent("Coventry City", KNOWN, client=client)
        system = client.last_kwargs["system"]
        assert any("Arsenal" in b["text"] for b in system)
        assert any(b.get("cache_control") for b in system)


class TestOutputIsValidatedNotTrusted:
    def test_a_match_to_an_unknown_club_is_downgraded(self):
        """The model supplies judgement about football, not facts about our
        schema. A club we do not have is a malformed answer, and storing it
        would write exactly the corruption this design prevents."""
        got = _validate(
            AgentResolution("X", "matched", "Real Madrid CF", 0.99, "sure"), KNOWN
        )
        assert got.decision == "uncertain"
        assert got.resolved_to is None
        assert "not a known club" in got.validation_note

    def test_a_match_naming_nothing_is_downgraded(self):
        got = _validate(AgentResolution("X", "matched", None, 0.9, ""), KNOWN)
        assert got.decision == "uncertain"

    def test_a_valid_match_passes_through(self):
        got = _validate(AgentResolution("Spurs FC", "matched", "Tottenham", 0.9, ""), KNOWN)
        assert got.decision == "matched"
        assert got.resolved_to == "Tottenham"
        assert got.validation_note == ""

    def test_a_new_entity_carrying_a_target_has_it_dropped(self):
        """Otherwise a caller reads the target and treats a 'new club' answer
        as a match."""
        got = _validate(
            AgentResolution("Sheffield Wednesday", "new_entity", "Sheffield United", 0.8, ""),
            KNOWN,
        )
        assert got.resolved_to is None
        assert "dropped" in got.validation_note

    def test_validation_runs_on_the_real_path(self):
        client = FakeClient(decision("matched", "Nonexistent United"))
        out = resolve_with_agent("Coventry City", KNOWN, client=client)
        assert out.decision == "uncertain"


class TestCost:
    def test_cached_tokens_are_billed_not_ignored(self):
        """input_tokens excludes cached input, so counting only that would
        understate every run — the roster is most of the prompt."""
        r = AgentResolution("x", "matched", "Arsenal", 1.0, "",
                            input_tokens=100, output_tokens=50,
                            cache_write_tokens=200, cache_read_tokens=800)
        expected = 100 * 5e-6 + 200 * 6.25e-6 + 800 * 0.5e-6 + 50 * 25e-6
        assert r.cost_usd == pytest.approx(expected)
        assert r.cost_usd > 100 * 5e-6 + 50 * 25e-6


@pytest.fixture
def engine(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'a.db'}", future=True)
    init_db(eng)
    with eng.begin() as c:
        c.execute(text("INSERT INTO teams (name, division) VALUES ('Arsenal','E0')"))
    return eng


def _bootstrap(names):
    return {
        "teams": [{"id": i + 1, "name": n} for i, n in enumerate(names)],
        "elements": [],
        "element_types": [],
    }


class TestApplyingDecisions:
    def test_new_entity_creates_the_club(self, engine):
        ingest_fpl(engine, data=_bootstrap(["Arsenal", "Coventry City"]))
        apply_resolution("Coventry City", "new_entity", engine=engine,
                         confidence=0.95, rationale="distinct club")
        with engine.connect() as c:
            assert c.execute(
                text("SELECT COUNT(*) FROM teams WHERE name='Coventry City'")
            ).scalar_one() == 1
        assert pending_resolutions(engine) == []

    def test_matched_links_without_creating_a_club(self, engine):
        ingest_fpl(engine, data=_bootstrap(["Arsenal", "Gunners XI"]))
        apply_resolution("Gunners XI", "matched", "Arsenal", engine=engine)
        with engine.connect() as c:
            assert c.execute(text("SELECT COUNT(*) FROM teams")).scalar_one() == 1
            assert c.execute(
                text("SELECT resolved_to FROM name_resolutions WHERE raw_name='Gunners XI'")
            ).scalar_one() == "Arsenal"

    def test_matching_to_a_club_we_do_not_have_is_refused(self, engine):
        ingest_fpl(engine, data=_bootstrap(["Arsenal", "Gunners XI"]))
        with pytest.raises(ValueError, match="unknown team"):
            apply_resolution("Gunners XI", "matched", "Real Madrid", engine=engine)

    def test_an_uncertain_decision_cannot_be_applied(self, engine):
        with pytest.raises(ValueError, match="refusing to apply"):
            apply_resolution("Whoever", "uncertain", engine=engine)


class TestProvenanceSurvives:
    def test_a_settled_decision_is_not_overwritten_by_a_later_ingest(self, engine):
        """The bug this guards against actually happened.

        Applying "new_entity" creates the club, so the next ingest resolves the
        same name by exact match and — without the WHERE clause on the upsert —
        replaced `agent:...` with `exact`, erasing the record that a judgement
        was ever made.
        """
        ingest_fpl(engine, data=_bootstrap(["Arsenal", "Coventry City"]))
        apply_resolution("Coventry City", "new_entity", engine=engine,
                         method="agent:resolve-agent-v0.1", confidence=0.95,
                         rationale="distinct club")

        ingest_fpl(engine, data=_bootstrap(["Arsenal", "Coventry City"]))

        with engine.connect() as c:
            method, why = c.execute(
                text("""SELECT method, rationale FROM name_resolutions
                        WHERE raw_name='Coventry City'""")
            ).one()
        assert method == "agent:resolve-agent-v0.1"
        assert why == "distinct club"
