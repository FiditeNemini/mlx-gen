import pytest
from twine.exceptions import TwineException

from mflux.release import pypi_publisher
from mflux.release.pypi_publisher import PyPIPublisher


class FakeSettings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _write_dist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "mlx_gen-0.1.0-py3-none-any.whl").write_bytes(b"wheel")


def test_pypi_upload_failure_is_fatal(tmp_path, monkeypatch):
    _write_dist(tmp_path, monkeypatch)
    monkeypatch.setattr(pypi_publisher, "Settings", FakeSettings)

    def fail_upload(settings, files):
        raise TwineException("403 authentication failed")

    monkeypatch.setattr(pypi_publisher.upload, "upload", fail_upload)

    with pytest.raises(RuntimeError, match="PyPI upload failed"):
        PyPIPublisher._upload_to_pypi(
            token="token",
            repository="pypi",
            display_name="PyPI",
            package_name="mlx-gen",
            version="0.1.0",
        )


def test_pypi_upload_already_exists_remains_idempotent(tmp_path, monkeypatch):
    _write_dist(tmp_path, monkeypatch)
    monkeypatch.setattr(pypi_publisher, "Settings", FakeSettings)

    def already_exists(settings, files):
        raise TwineException("File already exists")

    monkeypatch.setattr(pypi_publisher.upload, "upload", already_exists)

    PyPIPublisher._upload_to_pypi(
        token="token",
        repository="pypi",
        display_name="PyPI",
        package_name="mlx-gen",
        version="0.1.0",
    )
