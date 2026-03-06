"""Pydantic v2 models for structured LLM input and output.

These models serve two purposes:
    - mirascope ``format`` targets — the LLM is forced to return JSON
      matching these schemas (via OpenAI structured outputs).
    - Internal data transfer objects between pipeline stages.
"""

from pydantic import BaseModel, Field

# -- Step 0: Tool Ideation ---------------------------------------------------

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


class IdeationResult(BaseModel):
    """Batch response from the tool ideation step."""

    tools: list[ToolDescription]


# -- Step 1: Table of Contents -----------------------------------------------

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


# -- Step 2: Section Generation ----------------------------------------------

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


# -- Pipeline metadata -------------------------------------------------------

class DocumentRecord(BaseModel):
    """Tracks a generated document for reporting / validation."""

    tool_name: str
    document_type: str
    html_path: str
    toc_path: str
    total_sections: int
    issues_summary: list[str] = Field(default_factory=list)
