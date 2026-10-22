from policy.models.compliance_result import ComplianceViolation


def requires_human_approval(
    warnings: list[ComplianceViolation],
) -> bool:
    """Return whether compliance warnings require human approval.

    Current policy:
    - Any warning requires human approval.
    - No warnings means no human approval is required.
    """
    return len(warnings) > 0