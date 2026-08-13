# V2V robot to robotic dog

## Input

- source video: `/private/tmp/bernini_official_20260810/assets/testcases/v2v/source_case3.mp4`

<img src="input_sheet_preview.png" alt="Input contact sheet" width="100%" />

Full resolution: [input_sheet.png](input_sheet.png)

## Request

- task: `v2v`
- prompt: Replace the white humanoid robot standing on the dark reflective surface with a sleek robotic dog in the same position and scale, preserving the dark studio background, lighting, reflections, and camera framing. The new subject should be a futuristic four-legged mechanical dog with a white outer shell, black joint details, subtle glowing eyes, and articulated metal legs. Match the original motion by having the robotic dog perform a comparable animated pose sequence, with natural mechanical movement and consistent shadows and reflections on the floor.

## Expected result

- The humanoid robot is replaced by a robotic dog.
- The dog stays in the same position and scale.
- Studio lighting, reflections, and camera framing stay consistent.

## Actual result

- not yet manually reviewed
- inspect the mlx-gen output and compare it against the expected result above

## Run parameters

- seed: `42`
- steps: `40`
- guidance mode: `v2v_apg`
- output: `848x480`
- frames: `81`
- fps: `16`

## Reproduce

```bash
uv run python validation_outputs/bernini_r_1_3b_2026_08_10/official_parity/run_official_public_cases.py --official-root /private/tmp/bernini_official_20260810 --out-dir validation_outputs/bernini_r_1_3b_2026_08_11/head_v2v_case3_full_v2 --case v2v_case3 --seed 42 --steps 40 --max-condition-size 848 --low-ram
```

## Official reference output

<img src="official_sheet_preview.png" alt="Official reference contact sheet" width="100%" />

Full resolution: [official_sheet.png](official_sheet.png)

## mlx-gen output

<img src="mlx_sheet_preview.png" alt="mlx-gen output contact sheet" width="100%" />

Full resolution: [mlx_sheet.png](mlx_sheet.png)

## Artifacts

- output: `v2v_case3.mp4`
- metadata: `v2v_case3.metadata.json`
- input sheet: [input_sheet.png](input_sheet.png)
- official sheet: [official_sheet.png](official_sheet.png)
- mlx sheet: [mlx_sheet.png](mlx_sheet.png)
- initial noise: `initial_noise.npy` in the source validation run (not bundled)
- official output: `/private/tmp/bernini_official_20260810/assets/testcases/v2v/v2v_case3_out.mp4`
- source validation run: `/Users/albou/projects/gh/mlx-gen/validation_outputs/bernini_r_1_3b_2026_08_11/head_v2v_case3_full_v2/v2v_case3` (local harness only)
