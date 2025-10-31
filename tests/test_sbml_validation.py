import glob
from pathlib import Path

import pytest

libsbml = pytest.importorskip("libsbml", reason="python-libsbml is required for SBML validation")
from libsbml import readSBMLFromFile, LIBSBML_SEV_ERROR


def discover_sbml_files() -> list[str]:
    patterns = [
        "assets/sbml/*.xml",
        "test_cases/**/petab/*.xml",
    ]
    files: set[str] = set()
    for p in patterns:
        files.update(glob.glob(p, recursive=True))
    return sorted(files)


@pytest.mark.parametrize("sbml_path", discover_sbml_files())
def test_sbml_consistency(sbml_path: str):
    # Load and run consistency check
    doc = readSBMLFromFile(sbml_path)
    doc.checkConsistency()

    error_log = doc.getErrorLog()
    errors: list[str] = []
    for i in range(error_log.getNumErrors()):
        err = error_log.getError(i)
        if err.getSeverity() >= LIBSBML_SEV_ERROR:
            errors.append(err.getMessage())

    assert not errors, (
        f"SBML validation errors in {sbml_path}:\n" + "\n".join(f" - {m}" for m in errors)
    )


def test_discovery_not_empty():
    files = discover_sbml_files()
    assert len(files) > 0, "No SBML files found to validate"
