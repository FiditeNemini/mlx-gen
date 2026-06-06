import os
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.z_image.variants import ZImageTurbo
from mflux.utils.version_util import VersionUtil

PATH = "tests/4bit/"


def test_skip_quantization_component_ignores_stale_mflux_q_metadata(tmp_path):
    component_path = tmp_path / "vae"
    component_path.mkdir()
    mx.save_safetensors(
        str(component_path / "0.safetensors"),
        {"weight": mx.zeros((1,))},
        {
            "quantization_level": "8",
            "mflux_version": VersionUtil.get_mflux_version(),
        },
    )
    component = ComponentDefinition(
        name="vae",
        hf_subdir="vae",
        skip_quantization=True,
    )

    _, quantization_level, mflux_version = WeightLoader._load_component(tmp_path, component)

    assert quantization_level is None
    assert mflux_version == VersionUtil.get_mflux_version()


def test_mflux_format_reads_metadata_without_double_loading_first_shard(monkeypatch, tmp_path):
    component_path = tmp_path / "transformer"
    component_path.mkdir()
    first_shard = component_path / "0.safetensors"
    second_shard = component_path / "1.safetensors"
    metadata = {
        "quantization_level": "8",
        "mflux_version": VersionUtil.get_mflux_version(),
    }
    mx.save_safetensors(str(first_shard), {"a": mx.zeros((1,))}, metadata)
    mx.save_safetensors(str(second_shard), {"b": mx.ones((1,))}, metadata)

    original_load = mx.load
    calls = []

    def tracked_load(path, *args, **kwargs):
        calls.append((Path(path).name, kwargs.get("return_metadata")))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr("mflux.models.common.weights.loading.weight_loader.mx.load", tracked_load)

    weights, quantization_level, mflux_version = WeightLoader._try_load_mflux_format(component_path)

    assert quantization_level == 8
    assert mflux_version == VersionUtil.get_mflux_version()
    assert weights["a"].shape == (1,)
    assert weights["b"].shape == (1,)
    assert calls == [("0.safetensors", None), ("1.safetensors", None)]


def test_mflux_format_uses_index_instead_of_loading_extra_safetensors(monkeypatch, tmp_path):
    component_path = tmp_path / "transformer"
    component_path.mkdir()
    metadata = {
        "quantization_level": "8",
        "mflux_version": VersionUtil.get_mflux_version(),
    }
    mx.save_safetensors(str(component_path / "0.safetensors"), {"a": mx.zeros((1,))}, metadata)
    mx.save_safetensors(str(component_path / "unused-source.safetensors"), {"polluted": mx.ones((1,))}, metadata)
    (component_path / "model.safetensors.index.json").write_text(
        """
        {
          "metadata": {"quantization_level": "8", "mflux_version": "test"},
          "weight_map": {"a": "0.safetensors"}
        }
        """
    )

    original_load = mx.load
    calls = []

    def tracked_load(path, *args, **kwargs):
        calls.append(Path(path).name)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr("mflux.models.common.weights.loading.weight_loader.mx.load", tracked_load)

    weights, quantization_level, _ = WeightLoader._try_load_mflux_format(component_path)

    assert quantization_level == 8
    assert weights["a"].shape == (1,)
    assert "polluted" not in weights
    assert calls == ["0.safetensors"]


def test_skip_quantization_component_updates_without_quantizing(monkeypatch):
    quantize_calls = []

    class DummyModel:
        def __init__(self):
            self.updated = None
            self.strict = None

        def update(self, weights, strict=False):
            self.updated = weights
            self.strict = strict

    component = ComponentDefinition(name="vae", hf_subdir="vae", skip_quantization=True)
    weights = LoadedWeights(
        components={"vae": {"weight": mx.zeros((1,))}},
        meta_data=MetaData(quantization_level=8, mflux_version="test"),
    )
    model = DummyModel()
    monkeypatch.setattr(
        "mflux.models.common.weights.loading.weight_applier.nn.quantize",
        lambda *args, **kwargs: quantize_calls.append((args, kwargs)),
    )

    resolved_bits = WeightApplier.apply_and_quantize_single(
        weights=weights,
        model=model,
        component=component,
        quantize_arg=8,
    )

    assert resolved_bits is None
    assert model.updated == weights.components["vae"]
    assert model.strict is False
    assert quantize_calls == []


class TestModelSaving:
    @pytest.mark.slow
    def test_save_and_load_4bit_model(self):
        # Clean up any existing temporary directories from previous test runs
        TestModelSaving.delete_folder_if_exists(PATH)

        try:
            # given a saved quantized model (and an image from that model)
            modelA = ZImageTurbo(quantize=4)
            image1 = modelA.generate_image(
                seed=42,
                prompt="Luxury food photograph",
                num_inference_steps=2,
                height=368,
                width=640,
            )
            modelA.save_model(PATH)
            del modelA

            # Verify that the mflux version is correctly saved in the model's metadata
            _, quantization_level, mflux_version = WeightLoader._try_load_mflux_format(Path(PATH) / "vae")
            assert mflux_version == VersionUtil.get_mflux_version(), "mflux version not correctly saved in metadata"  # fmt: off
            assert quantization_level == 4, "quantization level not correctly saved in metadata"  # fmt: off

            # when loading the quantized model (also without specifying bits)
            modelB = ZImageTurbo(model_path=PATH)

            # then we can load the model and generate the identical image
            image2 = modelB.generate_image(
                seed=42,
                prompt="Luxury food photograph",
                num_inference_steps=2,
                height=368,
                width=640,
            )
            np.testing.assert_array_equal(
                np.array(image1.image),
                np.array(image2.image),
                err_msg="image2 doesn't match image1.",
            )

        finally:
            # cleanup
            TestModelSaving.delete_folder(PATH)

    @staticmethod
    def delete_folder(path: str) -> None:
        return shutil.rmtree(path)

    @staticmethod
    def delete_folder_if_exists(path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Deleted folder: {path}")
        else:
            print(f"Folder does not exist: {path}")
