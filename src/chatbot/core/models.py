"""Pydantic v2 models for structured LLM input and output.

These models serve two purposes:
    - mirascope ``format`` targets — the LLM is forced to return JSON
      matching these schemas (via OpenAI structured outputs).
    - Internal data transfer objects between pipeline stages.
"""

from pydantic import BaseModel, Field

# -- Tool Ideation ---------------------------------------------------


class ToolDescription(BaseModel):
    """A single fictional software tool invented by the LLM."""

    name: str = Field(description="Product name (e.g. 'VaultSync')")
    purpose: str = Field(description="One-sentence purpose statement")
    category: str = Field(description="Industry vertical (e.g. FinTech)")
    typical_user_base: str = Field(
        description="Target audience (e.g. 'Enterprise IT admins')"
    )
    assigned_doc_types: list[str] = Field(
        default_factory=list,
        description="Document types assigned post-ideation",
    )


class ToolIdea(BaseModel):
    """LLM-only tool payload produced by the ideation stage."""

    name: str = Field(description="Product name (e.g. 'VaultSync')")
    purpose: str = Field(description="One-sentence purpose statement")
    category: str = Field(description="Industry vertical (e.g. FinTech)")
    typical_user_base: str = Field(
        description="Target audience (e.g. 'Enterprise IT admins')"
    )


class IdeationResult(BaseModel):
    """Batch response from the tool ideation step."""

    tools: list[ToolDescription]


class IdeationLLMResult(BaseModel):
    """Strict schema used only for ideation structured output."""

    tools: list[ToolIdea]


# -- Table of Contents -----------------------------------------------

class TOCEntry(BaseModel):
    """A single node in the document outline (recursive).

    Supports arbitrary nesting depth — not hard-coded to 2 levels.
    """

    id: str = Field(description="Section identifier (e.g. '2.1.3')")
    title: str = Field(description="Section heading text")
    children: list["TOCEntry"] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        description="Nested sub-sections",
    )


TOCEntry.model_rebuild()


class TableOfContents(BaseModel):
    """Full table of contents for one document."""

    document_type: str
    tool_name: str
    sections: list[TOCEntry]


class TOCEntryLLM(BaseModel):
    """LLM-only TOC node schema with explicit required fields."""

    id: str = Field(description="Section identifier (e.g. '2.1.3')")
    title: str = Field(description="Section heading text")
    children: list["TOCEntryLLM"] = Field(  # pyright: ignore[reportUnknownVariableType]
        description="Nested sub-sections; use [] when leaf"
    )


TOCEntryLLM.model_rebuild()


class TableOfContentsLLM(BaseModel):
    """Strict schema used only for TOC structured output."""

    sections: list[TOCEntryLLM]


# -- Section Generation ----------------------------------------------

class SectionOutput(BaseModel):
    """LLM output for a single document section."""

    section_id: str = Field(description="Matches TOCEntry.id")
    html_content: str = Field(
        description="Generated HTML fragment for this section"
    )
    issues_applied: list[str] = Field(
        default_factory=list,
        description="Data quality issues injected in this section",
    )


class SectionLLMOutput(BaseModel):
    """Strict schema used only for section structured output."""

    html_content: str = Field(
        description="Generated HTML fragment for this section"
    )
    issues_applied: list[str] = Field(
        description="Data quality issues injected in this section"
    )


class SectionIssueManifestEntry(BaseModel):
    """Issue manifest entry for one generated section."""

    section_id: str
    issues_applied: list[str] = Field(default_factory=list)


class DocumentIssueManifest(BaseModel):
    """Per-document issue manifest artifact."""

    tool_name: str
    document_type: str
    total_issues: int
    sections: list[SectionIssueManifestEntry]


# -- Pipeline metadata -------------------------------------------------------

class DocumentRecord(BaseModel):
    """Tracks a generated document for reporting / validation."""

    tool_name: str
    document_type: str
    html_path: str
    toc_path: str
    issues_manifest_path: str
    total_sections: int
    issues_summary: list[str] = Field(default_factory=list)


class FileValidationResult(BaseModel):
    """Validation status for one file."""

    path: str
    file_type: str
    is_valid: bool
    errors: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """Validation summary across all generated artifacts."""

    total_files: int
    valid_files: int
    invalid_files: int
    html_files: int
    json_files: int
    results: list[FileValidationResult] = Field(default_factory=list)
