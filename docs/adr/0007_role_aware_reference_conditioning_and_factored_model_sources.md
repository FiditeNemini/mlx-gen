# ADR 0007: Role-Aware Reference Conditioning And Factored Model Sources

Status: Accepted.

## Context

Some generative video models accept media with different roles in one request. A source video
defines the clip to edit, while one or more reference images describe subjects, objects, or style
to inject. A reference image is not necessarily the first frame of the result, so counting every
image as a primary `image-to-video` input makes task inference wrong. Rejecting every request that
contains both images and video also prevents exact reference-guided video models from expressing
their real contract.

Bernini-R 1.3B adds this pressure now. It implements reference-to-video and reference-guided
video-to-video with packed Wan latent segments and dedicated renderer weights. Its official
repository is also larger than a bounded local storage budget because it repeats stock Wan text
encoder, tokenizer, and VAE components. Silently borrowing components from another cache would be
space-efficient but would make provenance and compatibility unverifiable.

## Decision

MLX-Gen distinguishes primary media from conditioning media in generation planning.

- `--image` and `--video` remain primary task-selection inputs. The existing fail-closed rule
  against mixing primary images and primary videos remains in force.
- Repeatable `--reference-image` is a typed conditioning role. It is counted separately and may
  coexist with a primary video only when the selected capability explicitly allows it.
- Reference-to-video, where references do not define a first frame, remains the public
  `text-to-video` task with reference-image conditioning. It must not be mislabeled as
  first-frame `image-to-video`.
- Reference-guided source-video editing remains `video-to-video`, consistent with ADR 0006.
- Capabilities disclose minimum and maximum reference-image counts. Unsupported reference
  combinations fail during plan or CLI validation, before model loading.

Models whose weights are distributed as repeated components may resolve a factored source set:

- every component source is explicit in the model config or caller input;
- each source is resolved with only the patterns required for that component;
- transformer, VAE, text encoder, and tokenizer compatibility is validated fail-closed before
  generation; when the scheduler is code-native rather than sourced, its solver and flow-shift
  semantics are validated against the selected model profile;
- runtime metadata records the effective component repositories or local roots and revisions;
- no missing component is silently substituted from an arbitrary cache.

Bernini is implemented as a dedicated Wan-family runtime because its packed source-aware RoPE and
three-/four-pass guidance semantics are not VACE control conditioning and are not stock Wan CFG.
Shared Wan blocks, VAE, text encoding, schedulers, lifecycle, callbacks, and output helpers should
be reused where their contracts are identical.

## Consequences

### Positive

- Task names describe the generated workflow rather than the file extensions present in a
  request.
- VACE, Bernini, and future exact routes can expose reference images without weakening the
  primary-media fail-closed boundary.
- Large composite checkpoints can reuse byte-identical or numerically compatible components with
  auditable provenance and bounded downloads.
- Dedicated renderer math remains isolated from ordinary Wan and VACE regression paths.

### Negative

- Capability schema consumers must understand a new additive reference-image count contract.
- Initializers and metadata need component-level source bookkeeping rather than one monolithic
  root path.
- Compatibility checks and proof artifacts add implementation and maintenance work.

### Neutral

- This ADR does not create `r2v`, `rv2v`, or `mv2v` public task names.
- Planner-backed motion rewriting may later improve prompts, but renderer-only prompt-guided
  `video-to-video` must not be advertised as a separately proven `mv2v` system.
- Factored loading does not authorize model upload, cache deletion, or redistribution.

## Enforcement

- `GenerationCapability` and `GenerationPlan` own reference-image count truth.
- Unified CLI routing must consume and re-emit reference images without adding them to primary
  `image_count`.
- Each model runtime validates its supported reference combinations again before weight-heavy
  work.
- Factored component resolution must use explicit source fields and required-pattern checks.
- Metadata and failure manifests preserve reference paths and component provenance.
- Model docs may claim a reference-guided route only after the ADR 0001 model-backed proof gate.

## Validation

- Planner and router tests cover zero, one, multiple, mixed source-video/reference, unsupported,
  and metadata-replay cases.
- Existing image/video task inference and ordinary Wan/VACE tests remain green.
- Factored-source tests prove exact component routing, missing/incompatible component rejection,
  and no unintended full-repository download pattern.
- Bernini numeric tests cover source-aware RoPE, heterogeneous segment packing, target extraction,
  scheduler state, and task-specific guidance algebra against the official reference.
- Real proof bundles contain source/reference assets, MP4s, contact sheets, sidecars, exact model
  revisions, commands, wall time, and whole-process memory measurements.

## Backlog links

- [0105 Bernini-R 1.3B renderer integration](../backlog/completed/0105_bernini_r_1_3b_renderer_integration.md)
- [0106 Bernini-R 1.3B full-trajectory parity and release quality](../backlog/planned/0106_bernini_full_trajectory_parity_and_release_quality.md)
- [0080 Wan2.1-VACE-1.3B native MLX port](../backlog/completed/0080_wan_vace_1_3b_native_port.md)

## Related

- [ADR 0001: Runtime Smoke Validation For Model Routes](0001_runtime_smoke_validation_for_model_routes.md)
- [ADR 0002: No Silent Automatic Fallbacks](0002_no_silent_automatic_fallbacks.md)
- [ADR 0003: Runtime Truth Versus Consumer Convenience](0003_runtime_truth_vs_consumer_convenience.md)
- [ADR 0006: Generative Video Editing Task Boundary](0006_generative_video_editing_task_boundary.md)
- [src/mflux/task_inference.py](../../src/mflux/task_inference.py)
- [src/mflux/models/wan/wan_initializer.py](../../src/mflux/models/wan/wan_initializer.py)
