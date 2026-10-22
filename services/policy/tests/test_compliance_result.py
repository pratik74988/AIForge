import pytest
from pydantic import ValidationError

from policy.models.compliance_result import (
    ComplianceResult,
    ComplianceViolation,
)


def test_compliance_violation_creation():
    violation = ComplianceViolation(
        code="UNSUPPORTED_MODEL",
        message="The requested model is not supported.",
    )

    assert violation.code == "UNSUPPORTED_MODEL"
    assert violation.message == "The requested model is not supported."


def test_approved_compliance_result():
    result = ComplianceResult(approved=True)

    assert result.approved is True
    assert result.errors == []
    assert result.warnings == []


def test_rejected_compliance_result():
    result = ComplianceResult(
        approved=False,
        errors=[
            ComplianceViolation(
                code="UNSUPPORTED_MODEL",
                message="The requested model is not supported.",
            )
        ],
    )

    assert result.approved is False
    assert len(result.errors) == 1
    assert result.errors[0].code == "UNSUPPORTED_MODEL"


def test_result_supports_warnings():
    result = ComplianceResult(
        approved=True,
        warnings=[
            ComplianceViolation(
                code="HIGH_RESOURCE_REQUEST",
                message="Requested resources are unusually high.",
            )
        ],
    )

    assert result.approved is True
    assert len(result.warnings) == 1
    assert result.warnings[0].code == "HIGH_RESOURCE_REQUEST"


def test_violation_requires_code():
    with pytest.raises(ValidationError):
        ComplianceViolation(
            code="",
            message="Some message",
        )


def test_violation_requires_message():
    with pytest.raises(ValidationError):
        ComplianceViolation(
            code="SOME_ERROR",
            message="",
        )

def test_compliance_result_defaults_to_no_human_approval():
    result = ComplianceResult(approved=True)

    assert result.requires_human_approval is False


def test_compliance_result_can_require_human_approval():
    result = ComplianceResult(
        approved=True,
        requires_human_approval=True,
    )

    assert result.requires_human_approval is True


def test_rejected_result_does_not_require_human_approval_by_default():
    result = ComplianceResult(
        approved=False,
        errors=[
            ComplianceViolation(
                code="INVALID_METRIC",
                message="Metric is invalid.",
            )
        ],
    )

    assert result.requires_human_approval is False