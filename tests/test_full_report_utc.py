"""CopasiFullReportUTCStep: one-shot full-report UTC on the engine-core contract."""
from importlib.resources import files

import pytest
from engine_core import UniformTimeCourseStep

from viva_copasi.core import build_core
from viva_copasi.processes import CopasiFullReportUTCStep


def _model() -> str:
    return str(files("viva_copasi.composites") / "repressilator.xml")


def _step():
    return CopasiFullReportUTCStep(core=build_core())


def test_subclasses_the_engine_core_contract():
    assert issubclass(CopasiFullReportUTCStep, UniformTimeCourseStep)
    # Fixed contract inputs come from the base.
    assert _step().inputs() == {
        "model_source": "string",
        "time": "float",
        "n_points": "integer",
    }
    # Keeps the numeric_result port type for composite wiring.
    assert _step().outputs() == {"result": "numeric_result"}


def test_emits_canonical_full_report_shape():
    out = _step().update({"model_source": _model(), "time": 10.0, "n_points": 11})
    r = out["result"]
    assert set(r) == {"time", "columns", "values"}
    assert len(r["time"]) == 11
    assert len(r["values"]) == 11
    assert all(len(row) == len(r["columns"]) for row in r["values"])
    assert all(isinstance(t, float) for t in r["time"])
    assert all(isinstance(v, float) for row in r["values"] for v in row)


def test_reports_species_and_reaction_fluxes_by_sbml_id():
    """The full report carries species AND reaction fluxes — more than species-only."""
    r = _step().update({"model_source": _model(), "time": 5.0, "n_points": 6})["result"]
    cols = set(r["columns"])
    # Repressilator species (SBML ids) are present...
    assert {"X", "Y", "Z"} <= cols
    # ...and there are strictly more columns than the 6 species (the fluxes).
    assert len(r["columns"]) > 6


def test_rejects_degenerate_n_points():
    with pytest.raises(ValueError, match="n_points must be >= 2"):
        _step().update({"model_source": _model(), "time": 5.0, "n_points": 1})
