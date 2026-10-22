from policy.approval_policy import requires_human_approval
from policy.models.compliance_result import ComplianceViolation


def test_no_warnings_do_not_require_human_approval():
    warnings = []

    assert requires_human_approval(warnings) is False


def test_any_warning_requires_human_approval():
    warnings = [
        ComplianceViolation(
            code="RESOURCE_WARNING",
            message="Resource usage is high.",
        )
    ]

    assert requires_human_approval(warnings) is True


def test_multiple_warnings_require_human_approval():
    warnings = [
        ComplianceViolation(
            code="WARNING_ONE",
            message="First warning.",
        ),
        ComplianceViolation(
            code="WARNING_TWO",
            message="Second warning.",
        ),
    ]

    assert requires_human_approval(warnings) is True