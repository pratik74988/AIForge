from planner.models.training_spec import TrainingSpec

from policy.models.compliance_result import ComplianceResult
from policy.validator import ComplianceValidator


def check_compliance(spec: TrainingSpec) -> ComplianceResult:
    """Validate a TrainingSpec against AIForge compliance policies."""
    validator = ComplianceValidator()
    return validator.validate(spec)