from pathlib import Path

import pytest

from viva_copasi.processes import (
    _model_path_resolution,
    _looks_like_model_content,
    _load_model_source,
)

# Vendored SBML model used to exercise the raw-text loading path (#15).
TEST_MODEL = str(Path(__file__).parent / 'fixtures' / 'BIOMD0000000012_url.xml')


def test_missing_relative_path_raises_with_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as e:
        _model_path_resolution("does_not_exist.cps")
    msg = str(e.value)
    assert "does_not_exist.cps" in msg
    assert str(tmp_path / "does_not_exist.cps") in msg
    assert str(tmp_path) in msg


def test_missing_absolute_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _model_path_resolution(str(tmp_path / "nope.xml"))


def test_url_passes_through():
    url = "https://example.org/model.xml"
    assert _model_path_resolution(url) == url


def test_existing_absolute_path_unchanged(tmp_path):
    f = tmp_path / "m.xml"
    f.write_text("<sbml/>")
    assert _model_path_resolution(str(f)) == str(f)


# ---------------------------------------------------------------------------
# Issue #15 — model_source may be raw SBML/COPASI text, not just a URL/path.
# ---------------------------------------------------------------------------

def test_content_detection_recognizes_xml_text():
    assert _looks_like_model_content("<?xml version='1.0'?><sbml/>")
    assert _looks_like_model_content("  \n<sbml xmlns='...'/>")  # leading ws
    assert _looks_like_model_content("<COPASI/>")


def test_content_detection_rejects_paths_and_urls():
    assert not _looks_like_model_content("model.xml")
    assert not _looks_like_model_content("/abs/path/model.cps")
    assert not _looks_like_model_content("relative/model.xml")
    assert not _looks_like_model_content("https://example.org/model.xml")


def test_load_model_source_from_text(tmp_path, monkeypatch):
    """Raw SBML text loads via the string loader — no FileNotFoundError even
    from a cwd where no such file exists (#15)."""
    monkeypatch.chdir(tmp_path)  # prove it is NOT treated as a path
    text = Path(TEST_MODEL).read_text(encoding='utf-8')
    dm = _load_model_source(text)
    assert dm is not None
    assert dm.getModel() is not None


def test_load_model_source_from_path_still_works():
    """A path still loads exactly as before (#15 keeps path/URL behavior)."""
    dm = _load_model_source(TEST_MODEL)
    assert dm is not None
    assert dm.getModel() is not None


def test_load_model_source_missing_path_still_raises(tmp_path):
    """A non-existent path still raises FileNotFoundError (not string-loaded)."""
    with pytest.raises(FileNotFoundError):
        _load_model_source(str(tmp_path / "nope.xml"))
