# ADS2V insert video on computer

## Input

- source video: `/private/tmp/bernini_official_20260810/assets/testcases/rv2v/source_case2.mp4`
- reference video 1: `/private/tmp/bernini_official_20260810/assets/testcases/rv2v/source_ref_case2.mp4`

![input](input_sheet.png)

## Request

- task: `ads2v`
- prompt: Add the video on the computer.

## Expected result

- The inserted video appears on the computer screen.
- The source scene stays intact outside the insertion region.
- The inserted content follows the screen area instead of taking over the whole scene.

## Actual result

- not yet manually reviewed
- inspect the mlx-gen output and compare it against the expected result above

## Run parameters

- seed: `42`
- steps: `40`
- guidance mode: `rv2v`
- output: `480x256`
- frames: `61`
- fps: `24`
- requested output target: `1280x672`, `61` frames, `24` fps

## Reproduce

```bash
uv run python validation_outputs/bernini_r_1_3b_2026_08_10/official_parity/run_official_public_cases.py --official-root /private/tmp/bernini_official_20260810 --out-dir validation_outputs/bernini_r_1_3b_2026_08_11/head_ads2v_mid480_61f --case ads2v --seed 42 --steps 40 --max-condition-size 480 --override-fps 24 --override-frames 61 --low-ram
```

## Official reference output

![official](official_sheet.png)

## mlx-gen output

![mlx-gen](mlx_sheet.png)

## Artifacts

- output: `validation_outputs/bernini_r_1_3b_2026_08_11/head_ads2v_mid480_61f/ads2v/ads2v.mp4`
- metadata: `validation_outputs/bernini_r_1_3b_2026_08_11/head_ads2v_mid480_61f/ads2v/ads2v.metadata.json`
- initial noise: `validation_outputs/bernini_r_1_3b_2026_08_11/head_ads2v_mid480_61f/ads2v/initial_noise.npy` (torch-cpu-manual-seed, seed=42)
- runtime policy: `low_ram=True`, `clear_cache_each_step=True`, `clear_cache_each_transformer_block=False`, `release_denoisers_before_decode=True`
- case json: `/private/tmp/bernini_official_20260810/assets/testcases/rv2v/rv2v_case2.json`
- official output: `/private/tmp/bernini_official_20260810/assets/testcases/rv2v/rv2v_case2_out.mp4`
