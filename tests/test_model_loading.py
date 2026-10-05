import pytest

from viva_copasi.processes import _model_path_resolution


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
