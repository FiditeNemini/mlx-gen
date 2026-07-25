from types import SimpleNamespace

import mlx.core as mx
import pytest

from mflux.models.wan.variants import Wan2_2_TI2V
from tests.wan.test_wan_a14b_config import _fake_t2v_a14b_model, _patch_fake_wan_generation


def _patch_grid_recording_scheduler(monkeypatch, recorded):
    class RecordingScheduler:
        num_train_timesteps = 1000

        def __init__(self, flow_shift):
            recorded["flow_shift"] = flow_shift
            self.timesteps = mx.array([], dtype=mx.int64)

        def set_timesteps(self, num_inference_steps=None, *, denoising_step_list=None):
            recorded["set_timesteps"] = {
                "num_inference_steps": num_inference_steps,
                "denoising_step_list": denoising_step_list,
            }
            if denoising_step_list is not None:
                self.timesteps = mx.array(denoising_step_list, dtype=mx.int64)
            else:
                self.timesteps = mx.array([900, 875], dtype=mx.int64)

        def step(self, model_output, timestep, sample, return_dict):
            return (sample,)

        def scale_noise(self, sample, timestep, noise=None):
            recorded.setdefault("scale_noise_calls", 0)
            recorded["scale_noise_calls"] += 1
            return sample

    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.WanUniPCMultistepScheduler", RecordingScheduler)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.WanEulerScheduler", RecordingScheduler)


def test_wan_generate_passes_step_grid_to_scheduler_and_metadata(monkeypatch):
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model, patch_to_video=False)
    recorded = {}
    observed = {}
    _patch_grid_recording_scheduler(monkeypatch, recorded)

    def to_video(**kwargs):
        observed["to_video"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video_from_frame_batches", to_video)
    timesteps_seen = []
    original_batch_timestep = Wan2_2_TI2V._batch_timestep
    monkeypatch.setattr(
        Wan2_2_TI2V,
        "_batch_timestep",
        staticmethod(
            lambda batch_size, timestep: (
                timesteps_seen.append(timestep) or original_batch_timestep(batch_size=batch_size, timestep=timestep)
            )
        ),
    )

    model.generate_video(
        seed=1,
        prompt="a slow wave",
        width=64,
        height=64,
        num_frames=1,
        guidance=1,
        guidance_2=1,
        denoising_step_list=[1000, 750, 500, 250],
    )

    assert recorded["set_timesteps"] == {
        "num_inference_steps": None,
        "denoising_step_list": [1000, 750, 500, 250],
    }
    # The denoise loop consumes the exact grid values.
    assert timesteps_seen == [1000, 750, 500, 250]
    # Metadata: steps records the grid length, flow_shift is honestly None
    # (never consulted), and the grid replays via --config-from-metadata.
    assert observed["to_video"]["steps"] == 4
    assert observed["to_video"]["flow_shift"] is None
    assert observed["to_video"]["extra_metadata"]["denoising_step_list"] == [1000, 750, 500, 250]


def test_wan_generate_default_step_count_unchanged_without_grid(monkeypatch):
    # No-flag pin: omitting num_inference_steps must still request the
    # historical 50-step schedule (None resolves to RECOMMENDED_STEPS).
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)
    recorded = {}
    _patch_grid_recording_scheduler(monkeypatch, recorded)

    model.generate_video(
        seed=1,
        prompt="a slow wave",
        width=64,
        height=64,
        num_frames=1,
        guidance=1,
        guidance_2=1,
    )

    assert recorded["set_timesteps"] == {"num_inference_steps": 50, "denoising_step_list": None}


def test_wan_generate_rejects_grid_with_explicit_step_count(monkeypatch):
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="mutually exclusive"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=1,
            num_inference_steps=4,
            guidance=1,
            guidance_2=1,
            denoising_step_list=[1000, 750, 500, 250],
        )


def test_wan_generate_rejects_grid_with_explicit_flow_shift(monkeypatch):
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="flow_shift"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=1,
            guidance=1,
            guidance_2=1,
            flow_shift=5.0,
            denoising_step_list=[1000, 750, 500, 250],
        )


def test_wan_generate_rejects_grid_with_video_to_video(monkeypatch):
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="not supported for Wan video-to-video"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=1,
            guidance=1,
            guidance_2=1,
            video_path="input.mp4",
            denoising_step_list=[1000, 750, 500, 250],
        )


def test_wan_generate_grid_boundary_routing_uses_grid_values(monkeypatch):
    # t2v A14B boundary is 0.875 * 1000 = 875: grid point 1000 runs the high
    # expert, 750/500/250 the low expert - the comparison consumes the exact
    # grid timesteps.
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)
    recorded = {}
    _patch_grid_recording_scheduler(monkeypatch, recorded)

    model.generate_video(
        seed=1,
        prompt="a slow wave",
        width=64,
        height=64,
        num_frames=1,
        guidance=1,
        guidance_2=1,
        denoising_step_list=[1000, 750, 500, 250],
    )

    assert len(model.transformer.calls) == 1
    assert len(model.transformer_2.calls) == 3


def test_wan_vace_rejects_step_grid():
    from mflux.models.wan.variants import WanVace

    model = WanVace.__new__(WanVace)

    with pytest.raises(ValueError, match="VACE does not support denoising_step_list"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            denoising_step_list=[1000, 750, 500, 250],
        )


def test_wan_generate_grid_scale_noise_not_involved_for_t2v(monkeypatch):
    # Sanity: the plain t2v grid path never calls scale_noise (no warm start),
    # so grid mode cannot silently interact with v2v machinery.
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)
    recorded = {}
    _patch_grid_recording_scheduler(monkeypatch, recorded)

    video = model.generate_video(
        seed=1,
        prompt="a slow wave",
        width=64,
        height=64,
        num_frames=1,
        guidance=1,
        guidance_2=1,
        denoising_step_list=[875, 500, 125],
    )

    assert video is not None
    assert recorded.get("scale_noise_calls", 0) == 0


def test_wan_generate_grid_boundary_equality_routes_high(monkeypatch):
    # Cycle-2 review pin: a grid value exactly ON the boundary (t2v 0.875 *
    # 1000 = 875) must route HIGH - the runtime uses the diffusers
    # `t >= boundary` convention, so [1000, 875, 500, 250] splits 2 high / 2
    # low, not 1/3.
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)
    recorded = {}
    _patch_grid_recording_scheduler(monkeypatch, recorded)

    model.generate_video(
        seed=1,
        prompt="a slow wave",
        width=64,
        height=64,
        num_frames=1,
        guidance=1,
        guidance_2=1,
        denoising_step_list=[1000, 875, 500, 250],
    )

    assert len(model.transformer.calls) == 2
    assert len(model.transformer_2.calls) == 2
