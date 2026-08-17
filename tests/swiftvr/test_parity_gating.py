"""The parity harness must skip when its inputs are absent, never fail and never fake.

A parity test that passes without the reference present would be worse than no test at
all, and one that *fails* on a machine with no 20 GB checkpoint makes the suite unusable
for everyone else. The contract is therefore: torch, the upstream source tree and each
checkpoint are skip conditions, checked by fixtures, with no synthetic stand-in anywhere.

This file verifies that contract by running the parity package in a subprocess with its
two roots redirected at an empty directory - the exact situation of a fresh clone - and
asserting the run is all skips and no failures.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.swiftvr.parity import parity_support

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PARITY_PACKAGE = Path("tests") / "swiftvr" / "parity"


# Tests inside the parity package that deliberately need neither torch nor the reference:
# they guard the harness itself rather than comparing against it, so they must keep running
# on a machine that has no checkpoint. Everything else in that package must skip.
REFERENCE_FREE_PARITY_TESTS = {
    "tests/swiftvr/parity/test_mfswa_parity.py::TestMfswaRoutingIsReal"
    "::test_real_weight_geometry_is_actually_shift_sensitive",
}


def _run_parity(env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src"), **env_overrides}
    return subprocess.run(
        # addopts is cleared so the inherited -m filter does not deselect the
        # high-memory parity tests: every one of them must be gated, not just the cheap ones.
        [sys.executable, "-m", "pytest", str(PARITY_PACKAGE), "-o", "addopts=", "-q", "--no-header", "-rp"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def _passed_node_ids(result: subprocess.CompletedProcess) -> set[str]:
    return {line.split(" ", 1)[1].strip() for line in result.stdout.splitlines() if line.startswith("PASSED ")}


class TestAvailabilityPredicates:
    def test_each_predicate_answers_with_a_bool(self):
        assert isinstance(parity_support.torch_available(), bool)
        assert isinstance(parity_support.reference_available(), bool)
        assert isinstance(parity_support.reae_weights_available(), bool)
        assert isinstance(parity_support.transformer_weights_available(), bool)

    def test_a_missing_checkpoint_is_reported_as_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(parity_support, "TRANSFORMER_CHECKPOINT", tmp_path / "absent.safetensors")
        assert parity_support.transformer_weights_available() is False
        assert parity_support.transformer_checkpoint_status().startswith("missing:")

    def test_a_partial_download_is_reported_as_incomplete(self, monkeypatch, tmp_path):
        """The failure mode worth guarding: the file opens, the header parses, and the
        error only surfaces once a tensor near the end is read."""
        partial = tmp_path / "diffusion_pytorch_model.safetensors"
        partial.write_bytes(b"\0" * 1024)
        monkeypatch.setattr(parity_support, "TRANSFORMER_CHECKPOINT", partial)
        assert parity_support.transformer_weights_available() is False
        status = parity_support.transformer_checkpoint_status()
        assert status.startswith("incomplete:")
        assert str(parity_support.TRANSFORMER_CHECKPOINT_BYTES) in status

    def test_a_missing_reae_checkpoint_is_reported(self, monkeypatch, tmp_path):
        monkeypatch.setattr(parity_support, "REAE_CHECKPOINT", tmp_path / "absent.safetensors")
        assert parity_support.reae_weights_available() is False

    def test_a_missing_reference_tree_is_reported(self, monkeypatch, tmp_path):
        monkeypatch.setattr(parity_support, "SWIFTVR_REFERENCE_ROOT", tmp_path)
        assert parity_support.reference_available() is False

    def test_the_reference_loader_refuses_rather_than_stubbing(self, monkeypatch, tmp_path):
        """No stub: a fabricated reference module would make the harness lie."""
        monkeypatch.setattr(parity_support, "SWIFTVR_REFERENCE_ROOT", tmp_path)
        parity_support.torch_reference.cache_clear()
        try:
            with pytest.raises(FileNotFoundError, match="reference source not found"):
                parity_support.torch_reference()
        finally:
            parity_support.torch_reference.cache_clear()


@pytest.mark.slow
class TestTheHarnessSkipsOnAFreshClone:
    def test_absent_inputs_produce_skips_and_no_failures(self, tmp_path):
        result = _run_parity(
            {
                "SWIFTVR_PARITY_REFERENCE": str(tmp_path / "no-reference"),
                "SWIFTVR_PARITY_SNAPSHOT": str(tmp_path / "no-snapshot"),
            }
        )
        summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        assert result.returncode == 0, f"parity suite did not exit cleanly:\n{result.stdout}\n{result.stderr}"
        assert "skipped" in summary, summary
        assert "failed" not in summary, summary
        assert "error" not in summary.lower(), summary

    def test_only_the_declared_reference_free_guards_run(self, tmp_path):
        """Every comparison against the reference must be gated by a fixture.

        A test that runs here without the reference present is measuring something other
        than parity. The two harness guards that legitimately need no reference are named
        in REFERENCE_FREE_PARITY_TESTS, so adding an ungated comparison fails this test
        instead of quietly passing on a machine with no checkpoint.
        """
        result = _run_parity(
            {
                "SWIFTVR_PARITY_REFERENCE": str(tmp_path / "no-reference"),
                "SWIFTVR_PARITY_SNAPSHOT": str(tmp_path / "no-snapshot"),
            }
        )
        assert _passed_node_ids(result) == REFERENCE_FREE_PARITY_TESTS, result.stdout
