# RV2V garment replacement

## Input

- source video: `/private/tmp/bernini_official_20260810/assets/testcases/rv2v/source_case1.mp4`
- reference image 1: `/private/tmp/bernini_official_20260810/assets/testcases/rv2v/ref_case1.jpg`

![input](input_sheet.png)

## Request

- task: `rv2v`
- prompt: Replace the person's outer shirt with the shirt from the reference image while keeping the inner undershirt unchanged, preserving the original body pose, fit behavior, camera framing, lighting, background, pants, hair, skin, shadows, and overall motion exactly as they are. The person stands against the same plain light gray studio backdrop with the same subtle movement and relaxed posture, still wearing the original yellow and white horizontally striped inner T-shirt underneath, while the outer garment is now a clean white button-up shirt with thin vertical dark pinstripes, a short stand collar, black front buttons, and a left chest pocket, appearing naturally worn on the body with realistic fabric drape and motion instead of hanging flat, and all other scene elements remain unchanged.

## Expected result

- The outer shirt is replaced by the white pinstripe button-up shirt.
- The inner yellow-and-white undershirt stays unchanged.
- Pose, body, background, lighting, and motion stay consistent.

## Actual result

- The outer shirt is replaced by the white pinstripe button-up while the person, pose, and scene stay intact.
- The shirt stays slightly more closed than in the official output, so less undershirt remains visible, but the fit and temporal stability are correct.
- Manual review on Tuesday, August 11, 2026 accepted this row as working.

## Run parameters

- seed: `42`
- steps: `20`
- guidance mode: `rv2v`
- output: `480x848`
- frames: `81`
- fps: `16`

## Reproduce

```bash
uv run python validation_outputs/bernini_r_1_3b_2026_08_10/official_parity/run_official_public_cases.py --official-root /private/tmp/bernini_official_20260810 --out-dir validation_outputs/bernini_r_1_3b_2026_08_10/official_parity_rv2v_case1_steps20_current_launchd_v1 --case rv2v_case1 --seed 42 --steps 20 --low-ram
```

## Official reference output

![official](official_sheet.png)

## mlx-gen output

![mlx-gen](mlx_sheet.png)

## Artifacts

- output: `validation_outputs/bernini_r_1_3b_2026_08_10/official_parity_rv2v_case1_steps20_current_launchd_v1/rv2v_case1/rv2v_case1.mp4`
- metadata: `validation_outputs/bernini_r_1_3b_2026_08_10/official_parity_rv2v_case1_steps20_current_launchd_v1/rv2v_case1/rv2v_case1.metadata.json`
- initial noise: `validation_outputs/bernini_r_1_3b_2026_08_10/official_parity_rv2v_case1_steps20_current_launchd_v1/rv2v_case1/initial_noise.npy` (torch-cpu-manual-seed, seed=42)
- runtime policy: `low_ram=True`, `clear_cache_each_step=True`, `clear_cache_each_transformer_block=False`, `release_denoisers_before_decode=True`
- case json: `/private/tmp/bernini_official_20260810/assets/testcases/rv2v/rv2v_case1.json`
- official output: `/private/tmp/bernini_official_20260810/assets/testcases/rv2v/rv2v_case1_out.mp4`
