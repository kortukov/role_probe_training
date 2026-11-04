"""Explicit compatibility for a verified pytest 3.3.1 parameter-ID difference.

Astropy7606 uses ids=str(unit). The dimensionless unit has an empty string;
pytest3.3.1 falls back to unit0, while benchmark metadata uses empty brackets.
Only rename this exact test node; never infer its verdict from suite totals.
Original test output remains preserved separately by the scorer.
"""
import re

def normalize_test_identity(sample_id: str, output: str) -> str:
    if sample_id != 'astropy__astropy-7606':
        return output
    old = 'astropy/units/tests/test_units.py::test_compose_roundtrip[unit0]'
    new = 'astropy/units/tests/test_units.py::test_compose_roundtrip[]'
    if new in output:
        return output
    return re.sub(re.escape(old) + r'(?=\s|$)', lambda match: new, output)
