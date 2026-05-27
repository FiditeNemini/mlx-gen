# Completed: Bonsai ternary FLUX.2 support

## Metadata

- Created: 2026-05-27
- Status: Completed
- Completed: 2026-05-27

## ADR status

- Governing ADRs: None
- ADR impact: Needs new ADR only if MLX-Gen introduces a general plugin/runtime-kernel provider
  boundary. No ADR is required for a narrow FLUX.2-compatible packed-loader path that follows
  existing model-family patterns.

## Context

Prism ML publishes Bonsai Image 4B Apple Silicon checkpoints as FLUX.2 Klein-derived text-to-image
models with a very small low-bit transformer, a 4-bit Qwen3 text encoder, and a BF16 Flux2 VAE.
The ternary checkpoint is `prism-ml/bonsai-image-ternary-4B-mlx-2bit`; the binary checkpoint is
`prism-ml/bonsai-image-binary-4B-mlx-1bit`. Both are cached locally under the Hugging Face cache.

## Current code reality

- `mlxgen generate --model prism-ml/bonsai-image-ternary-4B-mlx-2bit` now routes to the Bonsai
  text-to-image path and resolves the Bonsai repo id as a known FLUX.2 Klein-derived model.
- Ordinary FLUX.2 loading still expects `transformer/`, `text_encoder/`, and `vae/`
  subdirectories with standard MLX-Gen/mflux mapping. Bonsai is handled by an isolated
  pre-packed path for `transformer-packed-mflux/`, `text_encoder-mlx-4bit/`, `vae/`,
  `scheduler/`, and `tokenizer/`.
- Bonsai transformer weights are pre-packed `uint32` tensors with sibling BF16 `scales` and
  `biases`. The ternary repo declares `format: mlx-packed-affine`, `solver: ternary`, `bits: 2`,
  and `group_size: 128`.
- On this local MLX 0.31.0 runtime, a direct `mx.quantized_matmul` probe works for `bits=2,
  group_size=128`, but not for `bits=1, group_size=128`.
- Prism's public demo depends on `prism-image-studio` and `mflux-prism`. The useful part for
  MLX-Gen is the `mflux-prism` FLUX.2 `klein_fast` packed loader/megakernel path, not the demo app
  or server stack.
- MLX-Gen has a Bonsai-specific loader that reuses FLUX.2 Klein runtime semantics, the Qwen3 text
  encoder class, the Flux2 VAE, unified `mlxgen` routing, and the Prism-derived `klein_fast`
  packed transformer path.

## Problem

Bonsai is not a simple alias or ordinary q4/q8 prepared checkpoint. It is a FLUX.2-compatible
low-bit packed runtime artifact. Loading it through the standard `WeightLoader` would either miss
the subdirectories or try to treat packed `uint32` tensors as ordinary linear weights.

## Completed work

Added first-class support for the Bonsai ternary 2-bit MLX checkpoint while keeping the design
small: MLX-Gen reuses FLUX.2 pipeline semantics, adds only the packed transformer path required by
the Bonsai artifact, loads the existing 4-bit Qwen3 text encoder cleanly, and fails explicitly for
the binary 1-bit repo until the local MLX runtime has a supported 1-bit path.

## Why

Bonsai offers a high-value local Apple Silicon text-to-image model: a FLUX.2-like transformer with
much smaller disk and denoising memory footprint. It also validates MLX-Gen's strategy of
supporting real MLX-native model artifacts without forcing every model into q4/q8 prepared
folders.

## Requirements

- Detect Bonsai repo ids in `mlxgen` routing and model config resolution.
- Keep the user command simple, for example:
  `uv run mlxgen generate --model prism-ml/bonsai-image-ternary-4B-mlx-2bit --prompt ...`.
- Reuse FLUX.2 text-to-image semantics: FlowMatch Euler, guidance 1.0, no negative prompt, 4-step
  default for Bonsai.
- Add a packed transformer loader for `transformer-packed-mflux/` that consumes packed
  `weight/scales/biases` triples and BF16 skip tensors without converting them to ordinary
  dense weights.
- Load `text_encoder-mlx-4bit/` as the Qwen3 text encoder, preserving its stored 4-bit
  quantization rather than quantizing it again.
- Keep VAE and scheduler handling compatible with the FLUX.2/Flux2 VAE path.
- Produce helpful errors for the binary 1-bit repo if the active MLX runtime cannot execute
  `bits=1, group_size=128`.
- Keep Prism attribution and Apache 2.0 license handling visible in docs/model cards.

## Implementation summary

1. Added Bonsai aliases/family detection as a FLUX.2-derived text-to-image route.
2. Added a Bonsai variant and weight definition that know the Bonsai subdirectory names.
3. Ported the minimal Prism `mflux-prism` `klein_fast` packed loader and transformer wrapper into
   `src/mflux/models/flux2/model/flux2_transformer/klein_fast/`, isolated from ordinary FLUX.2.
4. Reused MLX `mx.quantized_matmul` for the ternary 2-bit path.
5. Added a runtime capability probe for low-bit packed affine support and turned unsupported
   binary 1-bit execution into a clear CLI error.
6. Added focused router and `prepare` tests, plus local image and timing validation against
   FLUX.2 Klein 4B q8.

## Scope

- Ternary Bonsai Image 4B MLX 2-bit text-to-image support.
- Routing, config, packed transformer loading, 4-bit Qwen3 text encoder loading, VAE/scheduler
  reuse, docs, and validation artifacts.

## Non-goals

- Do not implement Bonsai binary 1-bit in this item; track it separately until MLX 1-bit runtime
  support is available or a custom kernel decision is made.
- Do not vendor the full Prism demo app or `prism-image-studio` server stack.
- Do not add auto-download during generation.
- Do not publish derivative Bonsai weights under AbstractFramework unless license and attribution
  are explicitly rechecked.

## Dependencies and related tasks

- [Model integration roadmap](../planned/0001_model_integration_roadmap.md)
- [Bonsai binary 1-bit runtime support](../proposed/0004_bonsai_binary_1bit_runtime_support.md)
- `src/mflux/cli/mlx_gen.py`
- `src/mflux/models/common/config/model_config.py`
- `src/mflux/models/common/resolution/config_resolution.py`
- `src/mflux/models/common/weights/loading/weight_loader.py`
- `src/mflux/models/flux2/`
- Local Prism reference inspected at `/tmp/mflux-prism-inspect/src/mflux/models/flux2/model/flux2_transformer/klein_fast/`
- Local Bonsai cached snapshots under `/Users/albou/.cache/huggingface/hub/models--prism-ml--bonsai-image-*`

## Outcomes

- Ternary Bonsai can generate images through `mlxgen generate` without the user installing the
  Prism demo package.
- The implementation remains recognizably a FLUX.2 port on MLX, not a parallel application stack.
- Binary 1-bit fails with a short, truthful message rather than a crash or misleading support
  claim.
- Future AbstractVision integration can treat Bonsai as a supported local Apple Silicon T2I
  backend with clear capability metadata.

## Validation evidence

- `uv run mlxgen generate --model prism-ml/bonsai-image-ternary-4B-mlx-2bit --prompt "A bonsai
  tree in a quiet ceramic studio, soft morning light" --width 512 --height 512 --steps 4
  --guidance 1 --seed 42 --output validation_outputs/bonsai/ternary_512_seed42.png --metadata`
- `uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-4B -q 8 --prompt "A bonsai tree
  in a quiet ceramic studio, soft morning light" --width 512 --height 512 --steps 4 --guidance 1
  --seed 42 --output validation_outputs/bonsai/klein4b_q8_512_seed42.png --metadata`
- `uv run mlxgen generate --model prism-ml/bonsai-image-binary-4B-mlx-1bit --prompt "A bonsai
  tree" --width 512 --height 512 --steps 4 --guidance 1 --output
  validation_outputs/bonsai/binary_should_fail.png` fails with an explicit unsupported-1-bit
  message on MLX 0.31.0.
- `uv run pytest tests/cli/test_mlx_gen_router.py tests/cli/test_prepare_save.py -q`
- `uv run ruff check src/mflux/models/bonsai_image
  src/mflux/models/flux2/model/flux2_transformer/klein_fast
  src/mflux/models/flux2/variants/txt2img/flux2_klein.py src/mflux/cli/mlx_gen.py
  src/mflux/cli/defaults/defaults.py src/mflux/models/common/config/model_config.py
  src/mflux/models/common/cli/save.py
  src/mflux/models/common/weights/loading/weight_definition.py tests/cli/test_mlx_gen_router.py
  tests/cli/test_prepare_save.py`

Local comparison on the same prompt, seed 42, guidance 1, and 4 steps:

| Model | Disk footprint | 512px average time | 512px peak RSS | 1024px time | 1024px peak RSS | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bonsai ternary 2-bit | 3.6 GiB cached snapshot | 2.92 s | 3.57 GiB | 5.69 s | 3.60 GiB | Coherent; same visual family as Klein 4B q8. |
| FLUX.2 Klein 4B q8 | 22 GiB source cache, q8 applied at runtime | 3.55 s | 9.23 GiB | 6.81 s | 9.39 GiB | Coherent baseline; higher memory footprint. |
| Bonsai binary 1-bit | 3.2 GiB cached snapshot | Not runnable | Not runnable | Not runnable | Not runnable | Blocked by missing stock-MLX 1-bit packed affine runtime support. |

The 512px timing values are three cold-process repeats captured with `/usr/bin/time -l`.

## Progress checklist

- [x] Add Bonsai family detection and aliases.
- [x] Add Bonsai config resolution and defaults.
- [x] Add packed transformer loader for `transformer-packed-mflux/`.
- [x] Load `text_encoder-mlx-4bit/` without re-quantization.
- [x] Add low-bit runtime capability probe and binary 1-bit error path.
- [x] Generate and inspect at least one ternary image.
- [x] Update docs and model capability tables.

## Follow-up guidance

Do not promote binary 1-bit support until stock MLX can execute `bits=1, group_size=128` packed
affine matmuls or an ADR accepts a custom kernel/dependency path. Stock MLX 0.31.2 was checked on
2026-05-27 and still does not provide that runtime capability. Do not route Bonsai through
`mlxgen prepare`; these repositories are already packed runtime artifacts, so users should run
`mlxgen download` and `mlxgen generate`.
