"""Unit tests for issue planning logic."""

import pytest

from chatbot.core.models import TOCEntry
from chatbot.services.data_generation import issue_plan_manager as ipm


class _DeterministicRandom:
    """Simple deterministic random replacement for predictable tests."""

    def randint(self, _a: int, _b: int) -> int:
        return 2

    def sample(self, population: list[str] | tuple[str, ...], k: int) -> list[str]:
        return list(population)[:k]

    def choice(self, population: list[str] | tuple[str, ...]) -> str:
        return list(population)[0]


class TestIssuePlanManager:
    """Tests for IssuePlanManager behavior."""

    @staticmethod
    def _sections() -> list[TOCEntry]:
        return [
            TOCEntry(id="1", title="Intro", children=[]),
            TOCEntry(id="2", title="Scope", children=[]),
            TOCEntry(id="3", title="Terms", children=[]),
        ]

    def test_init_raises_when_toc_has_fewer_than_two_sections(self) -> None:
        # Arrange
        one_section = [TOCEntry(id="1", title="Intro", children=[])]

        # Act / Assert
        with pytest.raises(ValueError, match="need at least"):
            ipm.IssuePlanManager(one_section)

    def test_requirement_for_planned_section_returns_expected_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        monkeypatch.setattr(ipm.random, "Random", lambda: _DeterministicRandom())
        manager = ipm.IssuePlanManager(self._sections())

        # Act
        req = manager.requirement_for("1", issues_already_applied=0)

        # Assert
        assert manager.target_issue_total == 2
        assert req.required_issue_count == 1
        assert req.required_issue_label == "contradictory_clause"
        assert req.issues_remaining == 2

    def test_requirement_for_unplanned_section_is_clean(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        monkeypatch.setattr(ipm.random, "Random", lambda: _DeterministicRandom())
        manager = ipm.IssuePlanManager(self._sections())

        # Act
        req = manager.requirement_for("3", issues_already_applied=1)

        # Assert
        assert req.required_issue_count == 0
        assert req.required_issue_label is None
        assert req.issues_remaining == 1

    def test_issues_match_validates_both_clean_and_issue_sections(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        monkeypatch.setattr(ipm.random, "Random", lambda: _DeterministicRandom())
        manager = ipm.IssuePlanManager(self._sections())

        # Act
        planned_ok = manager.issues_match("1", ["contradictory_clause"])
        clean_ok = manager.issues_match("3", [])
        planned_bad = manager.issues_match("1", ["ambiguous_language"])
        clean_bad = manager.issues_match("3", ["minor_typo_or_formatting"])

        # Assert
        assert planned_ok is True
        assert clean_ok is True
        assert planned_bad is False
        assert clean_bad is False

    def test_validate_document_totals_raises_on_mismatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        monkeypatch.setattr(ipm.random, "Random", lambda: _DeterministicRandom())
        manager = ipm.IssuePlanManager(self._sections())

        # Act / Assert
        with pytest.raises(ValueError, match="issue plan expected"):
            manager.validate_document_totals(total_issues=1)
