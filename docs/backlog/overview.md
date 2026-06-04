# Backlog Overview

## Project summary

MLX-Gen is an independent Apple Silicon image and video generation package derived from
[mflux](https://github.com/filipstrand/mflux). The backlog tracks model integration work,
compatibility fixes, release-readiness work, and follow-up investigations that should survive
outside chat history.

## Counts

| State | Count |
| --- | ---: |
| Planned | 8 |
| Proposed | 6 |
| Completed | 5 |
| Deprecated | 0 |
| Recurrent | 1 |

## Next recommended work

1. Finish the [Wan video integrity release gate](planned/0016_wan_video_integrity_release_gate.md)
   before expanding any Wan A14B q8 full-size claim: wire health reports into release artifacts,
   add optional latent diagnostics, and revalidate exact full-size settings with the 0.18.9 guards.
2. Run [Wan prompt adherence parity validation](planned/0015_wan_prompt_adherence_parity_validation.md)
   before treating T2V/I2V prompt or motion behavior as quality-proven; explicitly match official
   Wan negative prompts and A14B guidance pairs in Diffusers-vs-MLX runs.
3. Validate and finish
   [Wan A14B boundary memory recovery and full-size validation](planned/0013_wan_a14b_boundary_memory_recovery.md)
   after the full-size I2V retry captures memory, exit-code, metadata, and output evidence across
   the high-noise to low-noise denoiser boundary.
4. Finish the [Wan quantization and motion parity](planned/0002_wan_quantization_motion_parity.md)
   residuals: validate I2V-A14B mixed q8 quality, decide q4 or mixed q4/q8 policy, and keep cards
   tied to exact passed settings.
5. Implement the
   [LoRA capability matrix and strict application](planned/0007_lora_capability_matrix_and_strict_application.md)
   item so user-requested adapters cannot be silently ignored after resolution.
6. Implement
   [first-class I2I modes and outpaint/reframe UX](planned/0019_first_class_i2i_modes_and_outpaint_reframe.md)
   so FLUX.1 Fill outpainting/reframing has a tested unified command/API that prepares the expanded
   canvas and mask internally.
7. Keep Bonsai binary 1-bit deferred in
   [proposed item 0004](proposed/0004_bonsai_binary_1bit_runtime_support.md) until stock MLX can
   execute the required 1-bit packed affine matmul or an ADR accepts a custom kernel path.
8. Investigate [Wan q8 performance](planned/0005_wan_q8_performance_investigation.md) only after
   integrity-gated outputs are healthy enough for timing claims; current public docs describe mixed
   q8/BF16 as storage and memory-footprint focused, not speed-improving.
9. Continue the [model integration roadmap](planned/0001_model_integration_roadmap.md) in priority
   order, starting with automated publication audits, supported q4/q8 validation, and
   gated-derivative hygiene.
10. Continue ERNIE-Image/Turbo after the Turbo text-to-image, Prompt Enhancer, and q4/q8
   validation work: add stronger Diffusers parity tests and non-turbo validation.
11. Continue Wan2.2 after the first TI2V-5B and A14B T2V/I2V milestones: add q8/q4 validation,
   stronger quality/performance checks, remaining cancel APIs, and keep SeedVR2 as the lower-risk
   existing-code video utility track.

## Planned ledger

| ID | Item | Area | Priority | Status |
| --- | --- | --- | --- | --- |
| 0001 | [Model integration roadmap](planned/0001_model_integration_roadmap.md) | Models, routing, quantization, UX | P0-P3 | Planned |
| 0002 | [Wan quantization and motion parity](planned/0002_wan_quantization_motion_parity.md) | Video, quantization, Diffusers parity | P0 | Planned |
| 0005 | [Wan q8 performance investigation](planned/0005_wan_q8_performance_investigation.md) | Video, performance, quantization | P1 | Planned |
| 0007 | [LoRA capability matrix and strict application](planned/0007_lora_capability_matrix_and_strict_application.md) | LoRA, routing, validation | P1 | Planned |
| 0013 | [Wan A14B boundary memory recovery and full-size validation](planned/0013_wan_a14b_boundary_memory_recovery.md) | Video, memory, progress, validation | P0 | Planned |
| 0015 | [Wan prompt adherence parity validation](planned/0015_wan_prompt_adherence_parity_validation.md) | Video, Diffusers parity, prompt adherence | P0 | Planned |
| 0016 | [Wan video integrity release gate](planned/0016_wan_video_integrity_release_gate.md) | Video, numerical integrity, release validation | P0 | Planned |
| 0019 | [First-class I2I modes and outpaint/reframe UX](planned/0019_first_class_i2i_modes_and_outpaint_reframe.md) | Image routing, I2I modes, FLUX Fill outpainting | P1 | Planned |

## Proposed ledger

| ID | Item | Area | Promotion criteria |
| --- | --- | --- | --- |
| 0004 | [Bonsai binary 1-bit runtime support](proposed/0004_bonsai_binary_1bit_runtime_support.md) | T2I, low-bit runtime | Promote after ternary works and MLX 1-bit packed affine runtime support is proven or accepted by ADR. |
| 0006 | [Wan I2V prompt motion validation](proposed/0006_wan_i2v_prompt_motion_validation.md) | Video, I2V quality | Promote only if planned item 0015 shows an I2V-specific motion or prompt-adherence gap that needs a separate fix. |
| 0008 | [Qwen edit parity expansion](proposed/0008_qwen_edit_parity_expansion.md) | Qwen edit, parity | Promote when Qwen edit quality/regression coverage becomes a release blocker. |
| 0009 | [Video second-family selection](proposed/0009_video_second_family_selection.md) | Video model roadmap | Promote after Wan stabilization leaves room for the next video backend. |
| 0010 | [LTX-2.3 conditioning and LoRA spike](proposed/0010_ltx2_conditioning_lora_spike.md) | Video, LTX, LoRA | Promote if LTX-2.3 becomes the selected second video family or a local spike proves feasibility. |
| 0011 | [Next-generation image/edit watchlist](proposed/0011_next_generation_image_edit_watchlist.md) | Image/edit roadmap | Promote when a watched model becomes locally cacheable, licensed, and useful enough for implementation. |

## Completed ledger

| ID | Item | Area | Completed | Outcome |
| --- | --- | --- | --- | --- |
| 0003 | [Bonsai ternary FLUX.2 support](completed/0003_bonsai_ternary_flux2_support.md) | T2I, FLUX.2, low-bit packed MLX | 2026-05-27 | Added Bonsai ternary 2-bit routing, packed transformer loading, q4 Qwen3 text-encoder loading, binary 1-bit runtime gating, docs, and local quality/speed validation against FLUX.2 Klein 4B q8. |
| 0012 | [Wan2.2 A14B T2V/I2V support](completed/0012_wan_a14b_t2v_i2v_support.md) | Video, Wan A14B, Diffusers parity | 2026-05-31 | Added T2V-A14B and I2V-A14B configs, dynamic two-transformer loading, Wan2.1-style VAE support, boundary routing, optional `--guidance-2`, fail-closed model identity checks, docs, tests, and MP4 smoke validation. |
| 0014 | [Shared progress callbacks for image and video pipelines](completed/0014_shared_progress_callbacks.md) | Callbacks, Python API, image/video progress | 2026-06-03 | Added one shared `ProgressEvent`, image progress subscriptions, Wan shared step-based progress events, CLI denoise-step progress, docs, and focused tests. |
| 0018 | [Taskless generation routing](completed/0018_taskless_generation_routing.md) | Routing, Python API, task validation | 2026-06-04 | Added public task inference, taskless T2I/I2I/T2V/I2V routing, FLUX.2 single-image I2I behavior, Wan A14B early fixed-task validation, generated-card/doc updates, and focused validation. |
| 0020 | [Generation capability contract and route planning](completed/0020_generation_capability_contract.md) | Routing, Python API, capabilities, I2I modes | 2026-06-04 | Added typed generation capabilities, public generation plans, `mlxgen capabilities`, `--i2i-mode`, metadata-aware route planning, local-path base-model hints, early option rejection, edit/I2I progress labeling, docs, and focused validation. |

## Deprecated ledger

No deprecated backlog items yet.

## Recurrent ledger

| ID | Item | Area | Cadence |
| --- | --- | --- | --- |
| 0017 | [Backlog release-state hygiene](recurrent/0017_backlog_release_hygiene.md) | Backlog, release state, follow-up triage | After each release, large validation run, or priority-changing item move. |

## Process

- New backlog items must use a unique four-digit global prefix.
- Planned items need current code reality, scope, non-goals, validation, and ADR status.
- New model routes must follow [ADR 0001](../adr/0001_runtime_smoke_validation_for_model_routes.md):
  do not mark support as working or release-ready without a model-backed smoke command and output
  evidence, or an intentional unsupported-state failure.
- Model, backend, architecture, task, and quantization resolution must follow
  [ADR 0002](../adr/0002_no_silent_automatic_fallbacks.md): ambiguous or unsupported requests fail
  closed unless fallback behavior is explicitly requested by the caller.
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
  quantized validation, cancellation APIs, and broader full-generation Diffusers parity. Shared
  progress callbacks were completed later in item 0014.
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
- Added 2026-05-30 Wan A14B priority item after confirming local T2V-A14B cache and then the full
  I2V-A14B cache. A14B is tracked separately from 5B quantization because it needs two-transformer
  boundary routing, scalar timesteps, `flow_shift=3.0`, and a Wan2.1-style VAE.
- Added 2026-05-30 ADR 0001 after the A14B route exposed a validation gap: new model routes now
  require model-backed smoke proof before support or release-readiness claims. The A14B T2V route
  has source-checkpoint MP4 smoke evidence; I2V-A14B has source-conditioned MP4 smoke evidence
  from the complete local I2V snapshot.
- Added 2026-05-30 ADR 0002 after Wan model resolution exposed silent default-config fallback risk:
  model identity, backend, architecture, task, and quantization resolution now fail closed unless a
  caller explicitly opts into fallback behavior.
- Completed 2026-05-31 Wan2.2 A14B T2V/I2V initial support. Remaining Wan work is tracked in the
  quantization, motion parity, and q8 performance items rather than in the A14B wiring item.
- Added 2026-06-02 Wan A14B boundary-memory item after a full-size I2V run appeared to stop at
  `denoise step 6/20`, which maps exactly to the high-noise/low-noise denoiser boundary for
  I2V-A14B at 20 steps.
- Added 2026-06-02 Wan T2V-A14B q8 validation: full q8 collapsed to near-black/static video, so
  Wan q8 now keeps `condition_embedder.*` and `proj_out` BF16. The rebuilt
  `models/wan2.2-t2v-a14b-diffusers-8bit` folder is about 40 GiB and passes the preserved
  384x224, 17-frame contact-sheet check, but short-run peak RSS did not improve versus BF16.
- Added 2026-06-02 Wan memory/speed follow-up: component-wise prepared-q8 loading produced a
  byte-identical sample but did not materially reduce the 19.5 GiB prepared-q8 peak RSS. The video
  conversion path now builds PIL frames one at a time to avoid full-video NumPy temporaries, which
  should matter most for 81/121-frame outputs.
- Added 2026-06-02 Wan package-level memory follow-up: RSS alone under-reported source-package
  memory pressure. MLX peak memory for the same 384x224 A14B T2V sample was 32.99 GiB from the
  upstream source package, 33.11 GiB from the BF16 prepared package, and 20.84 GiB from the mixed
  q8/BF16 prepared package. The BF16 prepared output was byte-identical to source; the mixed
  package is the first measured usage-memory reduction.
- Completed 2026-06-03 shared progress callbacks: image and video generation now use one
  lightweight `ProgressEvent`, image callers can subscribe through
  `model.callbacks.subscribe_progress(...)`, Wan still supports direct `progress_callback`, and
  CLI video progress advances by denoising step. Focused validation passed with 11 progress-related
  tests.
- Added 2026-06-04 first-class I2I mode and outpaint/reframe item after reviewing current routing
  semantics and FLUX Fill support. Public tasks should remain media-direction based, while
  latent img2img, edit-conditioned I2I, multi-reference I2I, inpainting, and outpainting become
  explicit internal modes or workflow options.
- Added 2026-06-03 Wan prompt-adherence parity item after source review confirmed MLX-Gen's Wan
  negative-prompt and A14B guidance defaults match official Wan recommended config semantics, while
  raw Diffusers omitted-argument defaults differ. No heavy validation was run because a long Wan
  generation was active.
- Added 2026-06-03 Wan video integrity release gate after a full-size T2V-A14B mixed q8/BF16 run
  completed after about 13h15m but encoded an all-black MP4 from non-finite decoded values. The q8
  A14B card must stay validation-sized until exact full-size integrity and quality validation passes;
  early tensor-health checks, CLI completion semantics, default video-health save validation, and
  failure manifests are now implemented.
- Ran 2026-06-04 post-release backlog hygiene after `mlx-gen` 0.18.9: promoted LoRA strictness to
  planned work because loader-level best-effort behavior conflicts with ADR 0002 and public docs,
  narrowed Wan A14B boundary-memory work to the remaining full-size I2V retry evidence, kept the
  Wan video integrity gate planned for release artifacts/full-size revalidation, and added recurrent
  backlog release-state hygiene.
- Completed 2026-06-04 taskless generation routing: `mlxgen generate` now infers common T2I/I2I/T2V/I2V
  tasks from model plus image inputs, keeps `--task` as an explicit override, rejects contradictory
  task/image shapes early, exposes public Python `resolve_task`/`infer_task`, and updates normal Wan
  docs and generated model-card usage snippets to omit unnecessary `--task`.
- Added 2026-06-04 generation capability contract item after image-edit routing review showed that
  `edit` must become an internal I2I mode, `--image-strength` must be restricted to latent img2img,
  and apps need a public capability/plan API before launching expensive generations.
- Completed 2026-06-04 generation capability contract: added typed capability descriptors,
  `GenerationPlan`, public `get_model_capabilities(...)` / `resolve_generation_plan(...)`,
  `mlxgen capabilities`, `--i2i-mode`, metadata-aware latent route replay, local-path
  `--base-model` handling, early option rejection, image edit progress labeling, docs, and
  lightweight route-plan artifacts. Focused resolver/router/progress validation passed with 86
  tests, then reviewer-driven metadata/base-model/mask/family-conflict coverage expanded the
  focused suite to 91 passing tests.
- Updated 2026-06-04 pre-0.18.10 backlog release state: current code has the taskless/capability
  planner in completed items 0018 and 0020, with docs now explicitly distinguishing latent
  img2img, edit/reference I2I, multi-reference I2I, `--image-strength`, and Wan video size rules.
  First-class outpaint/reframe remains planned in item 0019, and full-size Wan A14B q8 readiness
  remains gated by item 0016 rather than claimed in this release.
