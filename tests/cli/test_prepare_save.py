from unittest.mock import patch

import pytest

from mflux.models.common.cli import save


def test_prepare_wan_does_not_pass_lora_kwargs_to_non_lora_model(tmp_path, monkeypatch):
    observed = {}

    class FakeWan:
        def __init__(self, *, quantize, model_config):
            observed["init"] = {
                "quantize": quantize,
                "model_config": model_config.model_name,
            }

        def save_model(self, path):
            observed["path"] = path

    monkeypatch.setattr(save, "Wan2_2_TI2V", FakeWan)

    with patch(
        "sys.argv",
        [
            "mlxgen prepare",
            "--model",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "--path",
            str(tmp_path / "wan-q8"),
            "--quantize",
            "8",
        ],
    ):
        save.main()

    assert observed == {
        "init": {
            "quantize": 8,
            "model_config": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        },
        "path": str(tmp_path / "wan-q8"),
    }


def test_prepare_still_passes_lora_kwargs_to_lora_model(tmp_path, monkeypatch):
    observed = {}
    lora_path = tmp_path / "lora-a.safetensors"
    lora_path.write_bytes(b"")

    class FakeQwen:
        def __init__(self, *, quantize, model_config, lora_paths=None, lora_scales=None):
            observed["init"] = {
                "quantize": quantize,
                "model_config": model_config.model_name,
                "lora_paths": lora_paths,
                "lora_scales": lora_scales,
            }

        def save_model(self, path):
            observed["path"] = path

    monkeypatch.setattr(save, "QwenImage", FakeQwen)

    with patch(
        "sys.argv",
        [
            "mlxgen prepare",
            "--model",
            "Qwen/Qwen-Image",
            "--path",
            str(tmp_path / "qwen-q8"),
            "--quantize",
            "8",
            "--lora-paths",
            str(lora_path),
            "--lora-scales",
            "0.5",
        ],
    ):
        save.main()

    assert observed == {
        "init": {
            "quantize": 8,
            "model_config": "Qwen/Qwen-Image",
            "lora_paths": [str(lora_path)],
            "lora_scales": [0.5],
        },
        "path": str(tmp_path / "qwen-q8"),
    }


def test_prepare_rejects_prepacked_bonsai(tmp_path, capsys):
    with patch(
        "sys.argv",
        [
            "mlxgen prepare",
            "--model",
            "prism-ml/bonsai-image-ternary-4B-mlx-2bit",
            "--path",
            str(tmp_path / "bonsai"),
        ],
    ):
        with pytest.raises(SystemExit):
            save.main()

    assert "Bonsai checkpoints are already MLX-packed" in capsys.readouterr().err
