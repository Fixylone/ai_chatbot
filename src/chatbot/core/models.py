"""Pydantic v2 models for structured LLM output and pipeline data transfer."""

from pydantic import BaseModel, Field

# -- Tool Ideation ---------------------------------------------------


class ToolDescription(BaseModel):
    """Software tool with assigned document types."""

    name: str = Field(description="Product name (e.g. 'VaultSync')")
    purpose: str = Field(description="One-sentence purpose statement")
    category: str = Field(description="Industry vertical (e.g. FinTech)")
    typical_user_base: str = Field(
        description="Target audience (e.g. 'Enterprise IT admins')"
    )
    assigned_doc_types: list[str] = Field(default_factory=list)


class IdeationResult(BaseModel):
    """Batch response from the tool ideation step."""

    tools: list[ToolDescription]


class ToolIdeaResponse(BaseModel):
    """LLM-only tool description schema with explicit required fields."""

    name: str = Field(description="Product name (e.g. 'VaultSync')")
    purpose: str = Field(description="One-sentence purpose statement")
    category: str = Field(description="Industry vertical (e.g. FinTech)")
    typical_user_base: str = Field(
        description="Target audience (e.g. 'Enterprise IT admins')"
    )


class IdeationResponse(BaseModel):
    """Strict schema used only for ideation structured output."""

    tools: list[ToolIdeaResponse]


# -- Table of Contents -----------------------------------------------

class TOCEntry(BaseModel):
    """A single node in the document outline. Supports arbitrary nesting depth — not hard-coded to 2 levels."""

    id: str = Field(description="Section identifier (e.g. '2.1.3')")
    title: str = Field(description="Section heading text")
    children: list["TOCEntry"] = Field(
        default_factory=list,
        description="Nested sub-sections",
    )


TOCEntry.model_rebuild()


class TableOfContents(BaseModel):
    """Full table of contents for one document."""

    document_type: str
    tool_name: str
    sections: list[TOCEntry]


class TOCEntryResponse(BaseModel):
    """LLM-only TOC node schema with explicit required fields."""

    id: str = Field(description="Section identifier (e.g. '2.1.3')")
    title: str = Field(description="Section heading text")
    children: list["TOCEntryResponse"] = Field(
        description="Nested sub-sections; use [] when leaf"
    )


TOCEntryResponse.model_rebuild()


class TOCResponse(BaseModel):
    """Strict schema used only for TOC structured output."""

    sections: list[TOCEntryResponse]


# -- Section Generation ----------------------------------------------

class SectionOutput(BaseModel):
    """Internal representation of a generated HTML section."""

    section_id: str
    html_content: str
    issues_applied: list[str] = Field(default_factory=list)


class SectionResponse(BaseModel):
    """Strict schema used only for section structured output."""

    html_content: str = Field(
        description="Generated HTML fragment for this section"
    )
    issues_applied: list[str] = Field(
        description="Data quality issues injected in this section"
    )


class SectionIssueManifestEntry(BaseModel):
    section_id: str
    issues_applied: list[str] = Field(default_factory=list)


class DocumentIssueManifest(BaseModel):
    tool_name: str
    document_type: str
    total_issues: int
    sections: list[SectionIssueManifestEntry]


# -- Pipeline metadata -------------------------------------------------------


class DocumentRecord(BaseModel):
    tool_name: str
    document_type: str
    html_path: str
    toc_path: str
    issues_manifest_path: str
    total_sections: int
    issues_summary: list[str] = Field(default_factory=list)


class FileValidationResult(BaseModel):
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
    results: list[FileValidationResult] = []
