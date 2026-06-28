# Memory Reduction Validation Track

Items 0062-0064 still contain implemented memory changes that are not complete until real
quantitative evidence exists. Each remaining item must close with process-isolated before/after
statistics, performance numbers, and quality or continuity checks matching the affected model path.

Items 0059-0061 are completed with real process-isolated reports. Item 0059 proved the Wan streamed
decode/save path with eager-versus-streamed statistics and exact MP4 parity in
`validation_outputs/memory/real_generation_20260627_wan_r2/generation_memory_benchmark.json`.
Item 0060 proved runtime-memory telemetry overhead and parent physical-footprint agreement in
`validation_outputs/memory/real_generation_20260628_0060_zimage_runtime_telemetry_physical/generation_memory_benchmark.json`.
Item 0061 proved prompt-materialization peak RSS reductions with exact image parity in
`validation_outputs/memory/real_generation_20260628_0061_prompt_materialization_r3/generation_memory_benchmark.json`.

Item 0062 was reopened on 2026-06-28. The bounded `absolute_latent_frame` path reduced residency
only by changing restored output, and the 7B closure evidence used a frame-count scaling run rather
than a normal same-video 1:1 baseline/candidate test. Current normal package execution uses
`seedvr2_noise_mode=global`; the byte cap that can force bounded noise is internal benchmark-only.
The first 2026-06-28 normal 1:1 bundle was later rejected as production proof because it used
3B `25/8` and 7B `13/4` chunking. The corrected temporal-quality repair proof lives in
`validation_outputs/seedvr2_temporal_quality_repair_20260628/`, with the matching benchmark report
at
`validation_outputs/memory/seedvr2_temporal_quality_repair_20260628/generation_memory_benchmark.json`.
That bundle proves 3B and 7B restore the same 149-frame source to 149-frame `29/8` outputs with
all-boundary contact sheets and memory stats, but it does not close item 0062 because no
quality-preserving memory reduction has been proven.

Additional SeedVR2 1280px image evidence now lives in
`validation_outputs/memory/real_generation_20260627_seedvr2_image_1280_r2/generation_memory_benchmark.json`
and
`validation_outputs/memory/real_generation_20260627_seedvr2_image_1280_tiling_r2/generation_memory_benchmark.json`.
Those reports prove the fixed SeedVR2 image `--low-ram` path is stable and pixel-identical, while
the tuned explicit `--vae-tiling` path reduces MLX peak memory with a measured, non-zero pixel
delta. Keep `--vae-tiling` explicit unless a later product decision accepts that tradeoff.

The current acceptance rule is stricter than focused unit tests:

- record peak Darwin physical footprint or process RSS, MLX peak, MLX cache, and wall time;
- use separate fresh processes for baseline and candidate runs;
- keep exact model, seed, dimensions, frame count, steps, guidance, cache settings, and input
  artifacts in the benchmark report;
- include output health and quality comparison where model math should be unchanged;
- preserve generated artifacts and JSON summaries under `validation_outputs/memory/`.

Component-wise startup loading migration remains proposed as
[0065](../../proposed/0065_componentwise_weight_streaming_migration.md) until remaining item 0063
or another startup profile shows weight-loading overlap is still a dominant problem.
