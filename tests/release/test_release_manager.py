import pytest

from mflux.release import release_manager
from mflux.release.release_manager import ReleaseManager


@pytest.mark.fast
def test_release_manager_stops_before_git_artifacts_when_pypi_publish_fails(monkeypatch):
    calls = []

    monkeypatch.setattr(release_manager.VersionUtil, "get_mflux_version", lambda: "0.1.0")
    monkeypatch.setattr(release_manager.ReleaseValidator, "validate_release_ready", lambda version: None)
    monkeypatch.setattr(release_manager.GitOperations, "check_tag_exists", lambda tag_name: False)
    monkeypatch.setattr(
        release_manager.GitHubAPI,
        "check_github_release_exists",
        lambda github_token, github_repo, tag_name: False,
    )
    monkeypatch.setattr(release_manager.PyPIPublisher, "version_exists_on_pypi", lambda package_name, version: False)

    def build_and_verify_package():
        calls.append("build")

    def publish_to_pypi(pypi_token, package_name, version):
        calls.append("publish")
        raise RuntimeError("PyPI upload failed")

    def create_and_push_tag(tag_name, version):
        calls.append("tag")

    def create_github_release(github_token, github_repo, tag_name, version, release_notes):
        calls.append("github")

    monkeypatch.setattr(release_manager.PyPIPublisher, "build_and_verify_package", build_and_verify_package)
    monkeypatch.setattr(release_manager.PyPIPublisher, "publish_to_pypi", publish_to_pypi)
    monkeypatch.setattr(release_manager.GitOperations, "create_and_push_tag", create_and_push_tag)
    monkeypatch.setattr(release_manager.GitHubAPI, "create_github_release", create_github_release)

    with pytest.raises(RuntimeError, match="PyPI upload failed"):
        ReleaseManager.create_release(github_token="gh-token", pypi_token="pypi-token")

    assert calls == ["build", "publish"]
