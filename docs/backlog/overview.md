# Backlog Overview

## Project summary

MLX-Gen is an independent Apple Silicon image and video generation package derived from
[mflux](https://github.com/filipstrand/mflux). The backlog tracks model integration work,
compatibility fixes, release-readiness work, and follow-up investigations that should survive
outside chat history.

## Counts

| State | Count |
| --- | ---: |
| Planned | 3 |
| Proposed | 1 |
| Completed | 1 |
| Deprecated | 0 |
| Recurrent | 0 |

## Next recommended work

1. Finish the [Wan quantization and motion parity](planned/0002_wan_quantization_motion_parity.md)
   item: validate q8 quality, decide q4 or mixed q4/q8 policy, and keep video model cards honest
   before publishing Wan derivatives.
2. Keep Bonsai binary 1-bit deferred in
   [proposed item 0004](proposed/0004_bonsai_binary_1bit_runtime_support.md) until stock MLX can
   execute the required 1-bit packed affine matmul or an ADR accepts a custom kernel path.
3. Investigate [Wan q8 performance](planned/0005_wan_q8_performance_investigation.md), because
   the first same-settings q8 smoke run was substantially slower than BF16/source and did not
   capture peak memory.
4. Continue the [model integration roadmap](planned/0001_model_integration_roadmap.md) in priority
   order, starting with automated publication audits, supported q4/q8 validation, and
   gated-derivative hygiene.
5. Continue ERNIE-Image/Turbo after the Turbo text-to-image, Prompt Enhancer, and q4/q8
   validation work: add stronger Diffusers parity tests and non-turbo validation.
6. Continue Wan2.2 after the first text-to-video and first-frame image-to-video milestones: add
   q8/q4 validation, stronger quality/performance checks, progress/cancel APIs, and keep SeedVR2
   as the lower-risk existing-code video utility track.

## Planned ledger

| ID | Item | Area | Priority | Status |
| --- | --- | --- | --- | --- |
| 0001 | [Model integration roadmap](planned/0001_model_integration_roadmap.md) | Models, routing, quantization, UX | P0-P3 | Planned |
| 0002 | [Wan quantization and motion parity](planned/0002_wan_quantization_motion_parity.md) | Video, quantization, Diffusers parity | P0 | Planned |
| 0005 | [Wan q8 performance investigation](planned/0005_wan_q8_performance_investigation.md) | Video, performance, quantization | P1 | Planned |

## Proposed ledger

| ID | Item | Area | Promotion criteria |
| --- | --- | --- | --- |
| 0004 | [Bonsai binary 1-bit runtime support](proposed/0004_bonsai_binary_1bit_runtime_support.md) | T2I, low-bit runtime | Promote after ternary works and MLX 1-bit packed affine runtime support is proven or accepted by ADR. |

## Completed ledger

| ID | Item | Area | Completed | Outcome |
| --- | --- | --- | --- | --- |
| 0003 | [Bonsai ternary FLUX.2 support](completed/0003_bonsai_ternary_flux2_support.md) | T2I, FLUX.2, low-bit packed MLX | 2026-05-27 | Added Bonsai ternary 2-bit routing, packed transformer loading, q4 Qwen3 text-encoder loading, binary 1-bit runtime gating, docs, and local quality/speed validation against FLUX.2 Klein 4B q8. |

## Deprecated ledger

No deprecated backlog items yet.

## Process

- New backlog items must use a unique four-digit global prefix.
- Planned items need current code reality, scope, non-goals, validation, and ADR status.
- When implementation completes, move the item to `docs/backlog/completed/` and append completion
  evidence instead of deleting the planning history.
- If a backlog item establishes lasting architecture policy, create or update an ADR before
  closure.

## Planning notes

- Created the backlog system on 2026-05-25 while triaging local and online model integration
  candidates for MLX-Gen and AbstractVision.
- Refined the model roadmap on 2026-05-25 after checking Hugging Face model sizes, licenses,
  local cache state, and current T2I/I2I/T2V/I2V popularity.
- Added 2026-05-25 AbstractFramework publication audit note: checked Qwen, FLUX.2 Klein/Base,
  Z-Image, and Z-Image-Turbo q4/q8 repos are complete; future work should automate this audit.
- Added 2026-05-25 collection follow-up: several q8 and non-turbo Z-Image repos still need to be
  added to the Hugging Face `AbstractFramework / mlx-gen` collection once collection write
  permission is available.
- Added 2026-05-25 ERNIE follow-up: ERNIE Image Turbo text-to-image support exists with BF16,
  q8, q4, and optional Prompt Enhancer validation; remaining work is parity coverage and
  non-turbo ERNIE-Image.
- Added 2026-05-26 Wan follow-up: initial Wan2.2 TI2V text-to-video and first-frame
  image-to-video support exists with MP4 output, focused tests, and opt-in full-model parity
  fixtures for the Wan transformer, VAE encoder/decoder, prompt embeddings, scheduler replay, and
  a tiny latent-only CFG denoise loop; remaining work is decoded-video quality validation,
  quantized validation, progress/cancel APIs, and broader full-generation Diffusers parity.
- Added 2026-05-27 Wan focused item: fixed the q8 prepare blocker caused by unconditional LoRA
  kwargs, prepared and smoke-tested a q8 Wan folder locally, analyzed the user's three 121-frame
  videos for motion, and split remaining q8 quality/q4 policy work into planned item 0002.
- Added 2026-05-27 Wan q8 note: a same-settings 704x384, 25-frame, 12-step comparison found q8
  visually close to BF16/source but slower in that smoke run; peak runtime memory was not captured.
- Added 2026-05-27 Bonsai planning split: ternary 2-bit is planned as a FLUX.2-compatible packed
  loader; binary 1-bit is proposed/deferred because the local MLX runtime did not support the
  required 1-bit packed affine matmul probe.
- Added 2026-05-27 Wan performance item: q8 prepared output was visually close in the short
  comparison but took 217.4s versus 95.48s for source BF16, so a dedicated timing/memory
  investigation now tracks that abnormal slowdown.
- Completed 2026-05-27 Bonsai ternary 2-bit support: `mlxgen generate` can run
  `prism-ml/bonsai-image-ternary-4B-mlx-2bit` directly, binary 1-bit now fails with an explicit
  unsupported-runtime message, and a local validation panel compares Bonsai ternary with FLUX.2
  Klein 4B q8.
- Rechecked 2026-05-27 stock MLX 0.31.2 for Bonsai binary 1-bit. `bits=1, group_size=128`
  quantization still fails while 2/3/4/5/6/8 succeed, so proposed item 0004 remains deferred and
  the Bonsai ternary release can proceed.
