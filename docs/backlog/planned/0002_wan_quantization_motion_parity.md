# Planned: Wan quantization and motion parity

## Metadata

- Created: 2026-05-27
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: Needs new ADR only if Wan video progress/cancel/reporting becomes a stable
  AbstractVision provider API contract. No ADR is required for the current narrow prepare,
  validation, and Diffusers-parity fixes.

## Context

Wan2.2 TI2V is the first MLX-Gen text-to-video and first-frame image-to-video backend. It is also
large enough that small routing bugs, bad quantization policy, or subtle scheduler drift can waste
hours of local compute. A user reported that `mlxgen prepare` could not create a q8 Wan folder and
that three 121-frame, 50-step videos generated at 1280x704 felt too static.

## Current code reality

- The requested checkout path `/Users/albou/projects/gh/mlx-gen` does not exist on disk; the active
  MLX-Gen repository is `/Users/albou/projects/gh/sbx/mlx-gen`.
- `src/mflux/models/common/cli/save.py` previously passed `lora_paths` and `lora_scales` to every
  prepare backend. `Wan2_2_TI2V.__init__()` does not accept those kwargs, so Wan q8/q4 prepare
  failed before model loading or quantization.
- A local fix now inspects the selected model class signature and passes LoRA kwargs only to
  backends that declare `lora_paths`; `tests/cli/test_prepare_save.py` covers both Wan and Qwen.
- `WanWeightDefinition.quantization_predicate()` currently applies ordinary MLX quantization to
  any quantizable Wan module. No Wan-specific mixed q4/q8 policy has been validated yet.
- The local Wan source snapshot declares `WanPipeline`, `expand_timesteps: true`,
  `UniPCMultistepScheduler`, `flow_shift: 5.0`, `prediction_type: flow_prediction`, 30 transformer
  layers, and transformer `in_channels: 48`.
- MLX-Gen already has parity tests for Wan expanded timestep masks, scheduler timesteps/sigmas,
  scheduler replay, UMT5 prompt embeddings, transformer output, VAE encode/decode, and a tiny CFG
  denoise loop against Diffusers fixtures.
- On 2026-05-27, `mlxgen prepare --model Wan-AI/Wan2.2-TI2V-5B-Diffusers --quantize 8 --path
  models/wan2.2-ti2v-5b-diffusers-8bit` succeeded. The prepared folder is about 17 GiB:
  transformer 5.0 GiB, VAE 1.3 GiB, text encoder 11 GiB, tokenizer 16 MiB.
- Wan generated model cards now use `pipeline_tag: text-to-video`, Apache 2.0 frontmatter,
  video-generation tags, q8 Wan transformer/VAE wording, and a `video.mp4` usage example.
- A q8 prepared-folder smoke generation succeeded from that folder at 128x128, 5 frames, 2 steps,
  8 fps, guidance 1.0. This validates reload/wiring only, not quality.
- A same-settings 25-frame smoke comparison now exists for source BF16 versus the prepared q8
  folder at 704x384, 25 frames, 12 steps, 24 fps, guidance 5.0, seed 321. The source run took
  95.48 seconds; the prepared q8 run took 217.4 seconds. The q8 output stayed visually close in
  the contact sheet, but this run shows no speed win. Runtime peak memory was not captured by the
  Wan CLI/metadata path.
- The user-provided videos at `/Users/albou/Desktop/video-space.mp4`,
  `/Users/albou/Desktop/video-river.mp4`, and `/Users/albou/Desktop/video-long.mp4` are all
  1280x704, 121 frames, 24 fps, and about 5.04 seconds. Analysis artifacts live in
  `validation_outputs/wan/user_video_analysis/`.

## Problem

Wan q8 prepare had a concrete CLI/backend argument bug. Beyond that, MLX-Gen needs evidence that
its Wan implementation is not accidentally suppressing motion, that q8 keeps acceptable quality,
and that q4 should either remain unsupported, use full q4, or get a model-specific mixed q4/q8
policy.

## What we want to do

Make Wan video support publication-ready for AbstractFramework model repos and standalone users:
fix prepare, validate q8 reload and quality, compare motion behavior against Diffusers, decide q4
policy from evidence, and keep docs/backlog clear about what is verified versus experimental.

## Why

Wan runs are expensive. A 121-frame, 50-step generation at the recommended 1280x704 settings takes
about two hours on the user's M5 Max. Users need reliable progress, correct prepared folders, and
clear guidance before uploading or depending on quantized Wan checkpoints.

## Requirements

- `mlxgen prepare` must not pass unsupported LoRA kwargs to Wan.
- Prepared q8 Wan folders must reload and generate MP4 output.
- q8 quality must be compared with BF16/source at realistic settings before public model-card
  claims go beyond "loads and runs".
- Wan timing and memory reporting must be improved before performance claims: current metadata
  records generation time but not peak MLX memory, and the current q8 smoke comparison was slower
  than BF16/source.
- q4 must not be published as good unless side-by-side tests show acceptable quality. If full q4
  degrades motion or detail, define a mixed q4/q8 policy and document the retained q8/BF16 layers.
- Static-feeling outputs must be evaluated with frame contact sheets, frame-difference or optical
  flow metrics, and at least one Diffusers comparison using the same prompt/settings where
  feasible.
- Video model cards should identify `pipeline_tag: text-to-video` or image-to-video coverage, not
  inherit image-only wording.

## Suggested implementation

1. Keep the `save.py` signature-based LoRA forwarding fix and tests.
2. Add a Wan q8 prepared-folder quality panel at a documented lower-cost setting and, if compute
   allows, one 121-frame quality run.
3. Run the same prompt through upstream Diffusers from the local snapshot for a short comparison,
   or record why the hardware/runtime cost blocks it.
4. Prepare q4 only after enough disk is available, then compare BF16/q8/q4 with the same prompt,
   seed, dimensions, frames, steps, guidance, and fps.
5. If q4 fails quality, inspect layer sensitivity and implement a Wan-specific quantization
   predicate rather than reusing Qwen/ERNIE rules blindly.
6. Add or update generated Hugging Face cards for Wan q8/q4 once the quantization policy is known.

## Scope

- Wan2.2 TI2V 5B prepare, q8/q4 quantization validation, motion-quality checks, and docs/model-card
  readiness.
- Text-to-video and first-frame image-to-video only.

## Non-goals

- Do not upload AbstractFramework Wan q8/q4 repos until quality and model cards are verified.
- Do not port Wan A14B, Wan VACE, Wan Animate, or video-to-video in this item.
- Do not delete large local model folders automatically to recover disk; ask first.
- Do not treat q8 smoke tests at 128x128/5 frames as quality evidence.

## Dependencies and related tasks

- [Model integration roadmap](0001_model_integration_roadmap.md)
- `src/mflux/models/common/cli/save.py`
- `src/mflux/models/wan/variants/wan2_2_ti2v.py`
- `src/mflux/models/wan/scheduler/wan_unipc_multistep_scheduler.py`
- `src/mflux/models/wan/weights/wan_weight_definition.py`
- `tests/cli/test_prepare_save.py`
- `tests/wan/test_wan_local_parity.py`
- `tests/wan/test_wan_scheduler_and_timesteps.py`
- Local Diffusers reference: `/Users/albou/projects/gh/diffusers/src/diffusers/pipelines/wan/`
- Local Transformers reference: `/Users/albou/projects/gh/transformers/src/transformers/models/umt5/`

## Expected outcomes

- Wan q8 prepare and reload are no longer blocked by the LoRA argument bug.
- A future agent can see which Wan quantization levels are verified, which are only smoke-tested,
  and which remain unsafe to publish.
- Static or low-motion Wan outputs are evaluated against objective frame metrics and Diffusers
  parity instead of by impression alone.
- AbstractVision can depend on a clear Wan support state and progress-reporting behavior.

## Validation

- `uv run ruff check src/mflux/models/common/cli/save.py tests/cli/test_prepare_save.py`
- `uv run pytest tests/cli/test_prepare_save.py -q`
- `uv run pytest tests/wan/test_wan_scheduler_and_timesteps.py tests/wan/test_wan_quality_settings.py
  tests/wan/test_wan_progress.py -q`
- `MFLUX_RUN_LOCAL_WAN_PARITY=1 uv run pytest tests/wan/test_wan_local_parity.py -q`
- `uv run mlxgen prepare --model Wan-AI/Wan2.2-TI2V-5B-Diffusers --quantize 8 --path
  models/wan2.2-ti2v-5b-diffusers-8bit`
- `uv run mlxgen generate --model models/wan2.2-ti2v-5b-diffusers-8bit --task text-to-video
  --width 128 --height 128 --frames 5 --steps 2 --guidance 1 --fps 8 ...`
- Contact sheets and motion metrics for user videos under
  `validation_outputs/wan/user_video_analysis/`.

## Progress checklist

- [x] Identify the q8 prepare crash as unsupported LoRA kwargs passed into Wan.
- [x] Patch prepare backend instantiation to pass LoRA kwargs only to compatible model classes.
- [x] Add focused tests for Wan no-LoRA kwargs and Qwen LoRA preservation.
- [x] Verify Wan scheduler/timestep/progress tests.
- [x] Verify full local Wan parity fixtures against Diffusers.
- [x] Prepare a q8 Wan folder locally and measure size.
- [x] Fix generated Wan q8 model-card metadata and video usage.
- [x] Smoke-generate an MP4 from the prepared q8 folder.
- [x] Run a same-settings BF16/source versus prepared q8 short comparison.
- [x] Generate contact sheets and motion metrics for the three user-provided videos.
- [ ] Add Wan runtime peak-memory reporting to validation metadata or a documented validation
      harness.
- [ ] Add q8 quality comparison at publishable settings.
- [ ] Decide and validate Wan q4 or mixed q4/q8 policy.
- [ ] Update generated Wan model cards and public docs once quantization policy is settled.

## Guidance for the implementing agent

Re-check free disk before preparing more Wan folders. The prepared q8 folder consumed about 17 GiB
and only about 29 GiB remained afterward. Preserve user videos and generated model folders unless
the user explicitly authorizes cleanup.
