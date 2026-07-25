import mlx.core as mx
import numpy as np
import pytest

from mflux.models.wan.latent_creator import WanTimestepPolicy
from mflux.models.wan.scheduler import WanEulerScheduler, WanUniPCMultistepScheduler
from mflux.models.wan.variants import Wan2_2_TI2V


def test_wan_t2v_expanded_timesteps_match_diffusers_mask_policy():
    expanded = WanTimestepPolicy.expand_for_text_to_video(
        latent_shape=(2, 48, 3, 4, 6),
        timestep=937,
        patch_size=(1, 2, 2),
    )

    assert expanded.shape == (2, 18)
    assert np.all(np.array(expanded) == 937)


def test_wan_i2v_first_frame_timestep_mask_matches_diffusers_policy():
    mask = WanTimestepPolicy.first_frame_mask(latent_shape=(2, 48, 3, 4, 6))
    expanded = WanTimestepPolicy.expand_from_mask(mask=mask, batch_size=2, timestep=937, patch_size=(1, 2, 2))

    assert expanded.shape == (2, 18)
    expected = np.array([0] * 6 + [937] * 12, dtype=np.float32)
    np.testing.assert_array_equal(np.array(expanded[0]), expected)
    np.testing.assert_array_equal(np.array(expanded[1]), expected)


def test_wan_i2v_first_frame_condition_keeps_condition_frame_only():
    latents = mx.ones((1, 2, 3, 2, 2))
    condition = mx.zeros((1, 2, 3, 2, 2))
    first_frame_mask = WanTimestepPolicy.first_frame_mask(latent_shape=latents.shape)

    mixed = WanTimestepPolicy.apply_first_frame_condition(
        latents=latents,
        condition=condition,
        first_frame_mask=first_frame_mask,
    )

    mixed_np = np.array(mixed)
    assert np.all(mixed_np[:, :, 0] == 0)
    assert np.all(mixed_np[:, :, 1:] == 1)


def test_wan_unipc_flow_shift_5_timesteps_match_diffusers_reference():
    expected = {
        1: [999],
        2: [999, 833],
        3: [999, 909, 714],
        4: [999, 937, 833, 625],
        5: [999, 952, 882, 769, 556],
    }
    for steps, timesteps in expected.items():
        scheduler = WanUniPCMultistepScheduler()
        scheduler.set_timesteps(steps)
        assert np.array(scheduler.timesteps).tolist() == timesteps


def test_wan_unipc_flow_shift_5_sigmas_match_diffusers_reference():
    scheduler = WanUniPCMultistepScheduler()
    scheduler.set_timesteps(5)

    np.testing.assert_allclose(
        np.array(scheduler.sigmas),
        np.array(
            [
                0.9999989867210388,
                0.9524376392364502,
                0.8825258612632751,
                0.7696741223335266,
                0.55678790807724,
                0.0,
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_wan_unipc_order2_flow_prediction_steps_match_diffusers_reference():
    scheduler = WanUniPCMultistepScheduler()
    scheduler.set_timesteps(4)
    sample = mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 2, 2) / 10

    expected_sums = [
        27.45018768310547,
        26.876358032226562,
        25.032520294189453,
        19.023534774780273,
    ]
    expected_first_values = [
        [-0.006242090370506048, 0.09375791251659393, 0.19375790655612946],
        [-0.03015177696943283, 0.06984823197126389, 0.1698482185602188],
        [-0.10697832703590393, -0.006978313438594341, 0.09302167594432831],
        [-0.35735276341438293, -0.257352739572525, -0.1573527604341507],
    ]

    for index, timestep in enumerate(np.array(scheduler.timesteps).tolist()):
        model_output = mx.full(sample.shape, 0.1 * (index + 1), dtype=mx.float32)
        sample = scheduler.step(model_output, timestep, sample, return_dict=False)[0]
        np.testing.assert_allclose(float(mx.sum(sample).item()), expected_sums[index], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            np.array(sample.reshape(-1)[:3]),
            np.array(expected_first_values[index], dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
        )


def test_wan_euler_flow_shift_5_timesteps_match_lightx2v_reference():
    scheduler = WanEulerScheduler()
    scheduler.set_timesteps(4)

    np.testing.assert_allclose(
        np.array(scheduler.timesteps),
        np.array([1000.0, 937.5, 833.3333, 625.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(scheduler.sigmas),
        np.array([1.0, 0.9375, 0.8333333, 0.625, 0.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_wan_v2v_timesteps_follow_diffusers_strength_truncation():
    scheduler = WanUniPCMultistepScheduler(flow_shift=3.0)
    scheduler.set_timesteps(5)
    full = np.array(scheduler.timesteps).tolist()

    # strength 1.0 keeps the full schedule and starts at index 0.
    timesteps = Wan2_2_TI2V._video_to_video_timesteps(scheduler=scheduler, num_inference_steps=5, strength=1.0)
    assert timesteps == full
    assert scheduler.begin_index == 0

    # strength 0.7 at 5 steps truncates to floor(5 * 0.7) = 3 trailing steps.
    scheduler.set_timesteps(5)
    timesteps = Wan2_2_TI2V._video_to_video_timesteps(scheduler=scheduler, num_inference_steps=5, strength=0.7)
    assert timesteps == full[2:]
    assert scheduler.begin_index == 2

    # very small strength clamps to exactly one step at the final timestep.
    scheduler.set_timesteps(5)
    timesteps = Wan2_2_TI2V._video_to_video_timesteps(scheduler=scheduler, num_inference_steps=5, strength=0.01)
    assert timesteps == full[-1:]
    assert scheduler.begin_index == 4


def test_wan_unipc_scale_noise_uses_begin_index_sigma():
    scheduler = WanUniPCMultistepScheduler(flow_shift=3.0)
    scheduler.set_timesteps(5)
    sample = mx.ones((1, 2, 1, 2, 2), dtype=mx.float32)
    noise = mx.zeros_like(sample)

    scheduler.set_begin_index(0)
    scaled = scheduler.scale_noise(sample, scheduler.timesteps[0], noise)
    # sigmas[0] is deliberately 1 - 1e-6, so a strength-1.0 warm start keeps a 1e-6 source term.
    np.testing.assert_allclose(float(scaled[0, 0, 0, 0, 0].item()), 1.0 - float(scheduler.sigmas[0].item()), atol=1e-7)

    scheduler.set_timesteps(5)
    scheduler.set_begin_index(2)
    scaled = scheduler.scale_noise(sample, scheduler.timesteps[2], noise)
    np.testing.assert_allclose(float(scaled[0, 0, 0, 0, 0].item()), 1.0 - float(scheduler.sigmas[2].item()), atol=1e-7)


def test_wan_skips_high_noise_stage_detection():
    assert Wan2_2_TI2V._skips_high_noise_stage(timesteps=[900, 800], boundary_timestep=875.0) is False
    assert Wan2_2_TI2V._skips_high_noise_stage(timesteps=[800, 700], boundary_timestep=875.0) is True
    assert Wan2_2_TI2V._skips_high_noise_stage(timesteps=[800], boundary_timestep=None) is False
    assert Wan2_2_TI2V._skips_high_noise_stage(timesteps=[], boundary_timestep=875.0) is False


def test_wan_euler_steps_match_lightx2v_reference():
    scheduler = WanEulerScheduler()
    scheduler.set_timesteps(4)
    sample = mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 2, 2) / 10

    expected_sums = [
        27.44999885559082,
        26.950000762939453,
        25.450000762939453,
        19.450000762939453,
    ]
    expected_first_values = [
        [-0.00625, 0.09375, 0.19375001],
        [-0.02708334, 0.07291666, 0.17291667],
        [-0.08958334, 0.01041667, 0.11041667],
        [-0.33958334, -0.23958333, -0.13958333],
    ]

    for index, timestep in enumerate(np.array(scheduler.timesteps).tolist()):
        model_output = mx.full(sample.shape, 0.1 * (index + 1), dtype=mx.float32)
        sample = scheduler.step(model_output, timestep, sample, return_dict=False)[0]
        np.testing.assert_allclose(float(mx.sum(sample).item()), expected_sums[index], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            np.array(sample.reshape(-1)[:3]),
            np.array(expected_first_values[index], dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
        )


def test_wan_unipc_step_grid_matches_lightx2v_distill_contract():
    # 0099: the lightx2v Wan22StepDistillScheduler contract is the explicit
    # list [1000, 750, 500, 250]. The transformer must see EXACTLY these
    # timesteps; sigma follows the flow-matching identity t / 1000 with the
    # same leading 1e-6 guard the count path applies at sigma == 1 (the UniPC
    # lambda term log(1 - sigma) is degenerate there).
    scheduler = WanUniPCMultistepScheduler(flow_shift=5.0)
    scheduler.set_timesteps(denoising_step_list=[1000, 750, 500, 250])

    assert np.array(scheduler.timesteps).tolist() == [1000, 750, 500, 250]
    np.testing.assert_allclose(
        np.array(scheduler.sigmas),
        np.array([1.0 - 1e-6, 0.75, 0.5, 0.25, 0.0], dtype=np.float32),
        rtol=0,
        atol=1e-9,
    )
    assert scheduler.num_inference_steps == 4


def test_wan_euler_step_grid_matches_lightx2v_distill_contract():
    # The euler update tolerates sigma == 1 exactly, matching its own count
    # path (which also starts at sigma 1.0), so no leading clamp here.
    scheduler = WanEulerScheduler(flow_shift=5.0)
    scheduler.set_timesteps(denoising_step_list=[1000, 750, 500, 250])

    np.testing.assert_array_equal(
        np.array(scheduler.timesteps), np.array([1000.0, 750.0, 500.0, 250.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(np.array(scheduler.sigmas), np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32))


def test_wan_step_grid_ignores_flow_shift():
    # Grid entries are final (already-shifted) timesteps: two schedulers with
    # different flow_shift values must produce identical grid schedules.
    for scheduler_class in (WanUniPCMultistepScheduler, WanEulerScheduler):
        low_shift = scheduler_class(flow_shift=1.0)
        high_shift = scheduler_class(flow_shift=12.0)
        low_shift.set_timesteps(denoising_step_list=[875, 500, 125])
        high_shift.set_timesteps(denoising_step_list=[875, 500, 125])

        np.testing.assert_array_equal(np.array(low_shift.timesteps), np.array(high_shift.timesteps))
        np.testing.assert_array_equal(np.array(low_shift.sigmas), np.array(high_shift.sigmas))


def test_wan_step_grid_sigma_relation_matches_count_path_where_they_coincide():
    # The count path derives t = int(sigma * 1000); the grid inverts it as
    # sigma = t / 1000. Where both schedules announce the same timesteps
    # (count at flow_shift=1, steps=4 -> [999, 750, 500, 250]), the grid
    # sigmas agree within the count path's own timestep quantization error
    # (< 1/1000), and the clamped leading sigma is bitwise identical.
    count = WanUniPCMultistepScheduler(flow_shift=1.0)
    count.set_timesteps(4)
    count_timesteps = np.array(count.timesteps).tolist()
    assert count_timesteps == [999, 750, 500, 250]

    grid = WanUniPCMultistepScheduler(flow_shift=1.0)
    grid.set_timesteps(denoising_step_list=count_timesteps)

    assert np.array(grid.timesteps).tolist() == count_timesteps
    np.testing.assert_allclose(np.array(grid.sigmas), np.array(count.sigmas), rtol=0, atol=1.1e-3)
    # Non-clamped grid sigmas invert the count path's int map exactly.
    grid_sigmas = np.array(grid.sigmas)[:-1]
    assert (grid_sigmas[1:] * 1000).astype(np.int64).tolist() == count_timesteps[1:]


def test_wan_unipc_step_grid_uses_the_same_step_math_as_count_mode():
    # step() consumes only self.sigmas/self.timesteps and reset state, so a
    # count-mode scheduler forced onto the grid's schedule must reproduce the
    # grid run exactly: grid mode introduces no separate solver path.
    grid = WanUniPCMultistepScheduler()
    grid.set_timesteps(denoising_step_list=[1000, 750, 500, 250])
    forced = WanUniPCMultistepScheduler()
    forced.set_timesteps(4)
    forced.timesteps = grid.timesteps
    forced.sigmas = grid.sigmas

    sample_grid = mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 2, 2) / 10
    sample_forced = mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 2, 2) / 10
    for index, timestep in enumerate(np.array(grid.timesteps).tolist()):
        model_output = mx.full(sample_grid.shape, 0.1 * (index + 1), dtype=mx.float32)
        sample_grid = grid.step(model_output, timestep, sample_grid, return_dict=False)[0]
        sample_forced = forced.step(model_output, timestep, sample_forced, return_dict=False)[0]
        mx.eval(sample_grid, sample_forced)
        np.testing.assert_array_equal(np.array(sample_grid), np.array(sample_forced))
    assert bool(np.isfinite(np.array(sample_grid)).all())


@pytest.mark.parametrize("scheduler_class", [WanUniPCMultistepScheduler, WanEulerScheduler])
def test_wan_step_grid_rejects_malformed_grids(scheduler_class):
    scheduler = scheduler_class()
    for malformed in (
        [],
        [1000, 750, 800],  # not decreasing
        [1000, 750, 750],  # duplicate
        [1000, 0],  # below range
        [1001, 500],  # above range
        [1000, -5],  # negative
        [1000.0, 750.0],  # floats
        [True, False],  # bools
    ):
        with pytest.raises(ValueError):
            scheduler.set_timesteps(denoising_step_list=malformed)


@pytest.mark.parametrize("scheduler_class", [WanUniPCMultistepScheduler, WanEulerScheduler])
def test_wan_set_timesteps_takes_exactly_one_schedule_source(scheduler_class):
    scheduler = scheduler_class()
    with pytest.raises(ValueError, match="exactly one"):
        scheduler.set_timesteps()
    with pytest.raises(ValueError, match="exactly one"):
        scheduler.set_timesteps(4, denoising_step_list=[1000, 750])
