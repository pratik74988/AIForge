from pydantic import BaseModel, Field


class ComplianceViolation(BaseModel):
    """A deterministic compliance finding."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ComplianceResult(BaseModel):
    """Result produced by Compliance validation."""

    approved: bool
    requires_human_approval: bool = False
    errors: list[ComplianceViolation] = Field(default_factory=list)
    warnings: list[ComplianceViolation] = Field(default_factory=list)