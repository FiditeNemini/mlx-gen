import importlib
import types

from mflux.python_runtime import load_generation_model, resolve_generation_runtime


def test_python_runtime_selects_dedicated_bernini_renderer():
    runtime = resolve_generation_runtime(
        model="bernini-r-1.3b",
        reference_image_count=1,
        task="text-to-video",
    )

    assert runtime.runtime_id == "wan-bernini-r-1.3b"
    assert runtime.plan.capability_id == "bernini.reference-video"
    assert runtime.plan.reference_image_count == 1
    assert runtime._definition.import_path == "mflux.models.wan.variants.wan_bernini.BerniniRenderer"


def test_python_runtime_selects_bernini_reference_guided_video_edit():
    runtime = resolve_generation_runtime(
        model="bernini-r-1.3b",
        video_count=1,
        reference_image_count=2,
        task="video-to-video",
    )

    assert runtime.runtime_id == "wan-bernini-r-1.3b"
    assert runtime.plan.capability_id == "bernini.reference-video-edit"
    assert runtime.plan.video_count == 1
    assert runtime.plan.reference_image_count == 2


def test_python_loader_forwards_bernini_reference_images_at_generation(monkeypatch):
    original_import = importlib.import_module
    observed = {}

    class FakeBernini:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_video(self, **kwargs):
            observed["generate"] = kwargs
            return types.SimpleNamespace(task="text-to-video")

    def fake_import(name, package=None):
        if name == "mflux.models.wan.variants.wan_bernini":
            return types.SimpleNamespace(BerniniRenderer=FakeBernini)
        return original_import(name, package)

    monkeypatch.setattr("mflux.python_runtime.importlib.import_module", fake_import)

    loaded = load_generation_model(
        model="bernini-r-1.3b",
        reference_image_count=2,
        task="text-to-video",
        quantize=4,
    )
    loaded.generate_output(
        seed=7,
        prompt="put the referenced product on a marble table",
        reference_image_paths=["front.png", "side.png"],
    )

    assert observed["init"]["quantize"] == 4
    assert observed["generate"]["reference_image_paths"] == ["front.png", "side.png"]
    assert loaded.plan.reference_image_count == 2
