"""Issue planning service for section generation."""

from __future__ import annotations

import random
from dataclasses import dataclass

from chatbot.core.models import TOCEntry

MIN_DOCUMENT_ISSUES = 2
MAX_DOCUMENT_ISSUES = 3
ISSUE_LABELS = (
    "contradictory_clause",
    "ambiguous_language",
    "minor_typo_or_formatting",
    "inconsistent_terminology_or_cross_reference",
)


@dataclass(frozen=True)
class SectionIssueRequirement:
    """Issue requirements for one section generation request.

    Attributes:
        target_issue_total: Planned total issue count for the document.
        issues_remaining: Remaining issues needed after previous sections.
        required_issue_count: Number of issues required in this section.
        required_issue_label: Required issue label when one issue is required.
    """

    target_issue_total: int
    issues_remaining: int
    required_issue_count: int
    required_issue_label: str | None


class IssuePlanManager:
    """Own issue planning and validation for one generated document.

    The manager creates a random 2-3 issue plan and validates model outputs
    against per-section expectations.
    """

    def __init__(self, ordered_sections: list[TOCEntry]) -> None:
        """Initialize the issue plan for a document.

        Args:
            ordered_sections: Flattened section list in generation order.

        Raises:
            ValueError: If the TOC has fewer than two sections.
        """
        self._section_ids = [section.id for section in ordered_sections]
        if len(self._section_ids) < MIN_DOCUMENT_ISSUES:
            msg = (
                f"TOC has {len(self._section_ids)} sections; need at least "
                f"{MIN_DOCUMENT_ISSUES} sections to distribute document issues."
            )
            raise ValueError(msg)

        rng = random.Random()
        max_total = min(MAX_DOCUMENT_ISSUES, len(self._section_ids))
        self._target_issue_total = rng.randint(MIN_DOCUMENT_ISSUES, max_total)

        planned_sections = rng.sample(self._section_ids, k=self._target_issue_total)
        if self._target_issue_total <= len(ISSUE_LABELS):
            planned_labels = list(rng.sample(ISSUE_LABELS, k=self._target_issue_total))
        else:
            planned_labels = [
                rng.choice(ISSUE_LABELS) for _ in range(self._target_issue_total)
            ]
        self._issue_plan = dict(zip(planned_sections, planned_labels))

    @property
    def target_issue_total(self) -> int:
        """Return planned issue count for this document."""
        return self._target_issue_total

    def requirement_for(
        self,
        section_id: str,
        issues_already_applied: int,
    ) -> SectionIssueRequirement:
        """Build section-level requirement values for prompts and checks.

        Args:
            section_id: Current section identifier.
            issues_already_applied: Issues already accepted in prior sections.

        Returns:
            Requirement object for the current section.
        """
        required_issue_label = self._issue_plan.get(section_id)
        required_issue_count = 1 if required_issue_label else 0
        issues_remaining = max(0, self._target_issue_total - issues_already_applied)
        return SectionIssueRequirement(
            target_issue_total=self._target_issue_total,
            issues_remaining=issues_remaining,
            required_issue_count=required_issue_count,
            required_issue_label=required_issue_label,
        )

    def issues_match(self, section_id: str, issues_applied: list[str]) -> bool:
        """Check whether model issues match the plan for one section.

        Args:
            section_id: Current section identifier.
            issues_applied: Labels returned by the model for this section.

        Returns:
            True when issues exactly match planned requirement for the section.
        """
        required_issue_label = self._issue_plan.get(section_id)
        if required_issue_label is None:
            return len(issues_applied) == 0
        if len(issues_applied) != 1:
            return False
        return issues_applied[0].strip() == required_issue_label

    def validate_document_totals(self, total_issues: int) -> None:
        """Validate final document issue totals against plan and hard bounds.

        Args:
            total_issues: Total accepted issues across generated sections.

        Raises:
            ValueError: If totals violate planned target or allowed range.
        """
        if total_issues != self._target_issue_total:
            msg = (
                f"Document has {total_issues} issues, but issue plan expected "
                f"{self._target_issue_total}."
            )
            raise ValueError(msg)

        if total_issues < MIN_DOCUMENT_ISSUES or total_issues > MAX_DOCUMENT_ISSUES:
            msg = (
                f"Document has {total_issues} injected issues; expected "
                f"{MIN_DOCUMENT_ISSUES}-{MAX_DOCUMENT_ISSUES}."
            )
            raise ValueError(msg)
