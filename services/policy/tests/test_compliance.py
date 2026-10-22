from policy.compliance import check_compliance

from policy.tests.test_validator import make_valid_spec


def test_check_compliance_accepts_valid_spec():
    spec = make_valid_spec()

    result = check_compliance(spec)

    assert result.approved is True
    assert result.requires_human_approval is False
    assert result.errors == []
    assert result.warnings == []


def test_check_compliance_rejects_invalid_spec():
    spec = make_valid_spec()
    spec.model.algorithm = "unsupported_algorithm"

    result = check_compliance(spec)

    assert result.approved is False
    assert result.requires_human_approval is False
    assert len(result.errors) > 0


def test_check_compliance_requires_human_approval():
    spec = make_valid_spec()
    spec.resources.cpu = 12

    result = check_compliance(spec)

    assert result.approved is True
    assert result.requires_human_approval is True
    assert len(result.warnings) > 0