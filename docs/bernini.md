# Bernini-R 1.3B

MLX-Gen supports ByteDance's Bernini-R 1.3B renderer for reference-guided video generation and
editing on Apple Silicon. The integration is deliberately renderer-only: it exposes the useful
reference and source-video paths without bundling the optional Qwen2.5-VL semantic planner.

> **Experimental — not yet promoted.** The original 2026-08-04 release-quality bundle failed, but
> the official public 1.3B parity pass now has accepted qualitative evidence for `i2i`, `t2i`,
> `t2v`, `r2v`, `rv2v_case1`, `v2v_case1`, and `mv2v` (the last with three named caveats), plus
> `ads2v` accepted at mid profile (the full official recipe is unverifiable on this host for
> either engine). The two remaining rows fail at their official recipes identically in both
> implementations (oracle-proven, so not mlx-gen defects), but documented non-official recipes
> recover them substantially: `v2v_case3` achieves the full quadruped robotic dog with the mv2v
> task prefix + guidance 5.0, and `r2v_case2` recovers cat-ear elements, chest branding, and the
> drinking motion with reference guidance 6.0 (seed-sensitive). Do not use this route for
> production work yet.

Upstream public Bernini examples are broader than the currently proven mlx-gen surface. The strict
official-example target is tracked separately in the
[official public 1.3B example parity matrix](assets/validation/bernini-r-1.3b-2026-08-04/official_example_parity_matrix.md).

As of Wednesday, August 12, 2026, the accepted current official-case rows are:

- `i2i` — accepted with a minor wall-texture caveat against the official image
- `t2i`
- `t2v`
- `r2v`
- `rv2v_case1` — accepted with a prompt-fidelity caveat: the shirt stays more closed than the
  official example, so less undershirt is exposed, but the garment replacement, fit, and temporal
  stability are qualitatively good
- `v2v_case1` — snowman insertion, accepted on 2026-08-11 review
- `mv2v` (`v2v_case2`) — crouch edit, accepted with three named caveats (invented standing start
  pose with cropped head at frame 0, later crouch onset than the retained clip, and the dog
  standing late in the clip against the prompt's "stays seated" constraint)

`ads2v` is accepted at mid profile: implementation parity is proven against the official code
(near frame-level trajectory match at identical settings and noise) and the mid-profile artifact
shows the inserted ad playing through its "Bernini" logo reveal; the full official recipe is
unverifiable on this host for either engine. `v2v_case3` and `r2v_case2` fail at their official
recipes in both implementations (oracle-proven — the official implementation produces the same
or worse results at identical settings), while documented non-official recipes recover them
substantially (quadruped dog via the mv2v prefix; most reference properties via reference
guidance 6.0). The retained upstream example clips are almost
certainly 14B output (upstream's scripts default to the 14B checkpoint and its README says the
1.3B "lags behind on complex tasks"). See the
[official public 1.3B example parity matrix](assets/validation/bernini-r-1.3b-2026-08-04/official_example_parity_matrix.md)
for the full evidence trail, including the local official-code oracle methodology.

Use Bernini when ordinary images should act as reusable visual references, or when a source video
should be edited with those references. Continue using Wan VACE for learned mask/control workflows
and Wan A14B for general text/image-to-video generation.

## Supported Workflows

| Inputs | MLX-Gen mode | What it does |
| --- | --- | --- |
| 1-8 `--reference-image` values | R2V (`reference-video`) | Generates a new video from ordered ordinary reference images and a prompt. |
| One `--video` plus 1-8 references | RV2V (`reference-video-edit`) | Edits the source clip while using the images as appearance/content references. |
| One `--video`, no references | V2V (`latent-video`) | Performs prompt-guided source-video editing or motion change. |

The renderer does not currently expose Bernini's planner, T2I/I2I convenience routes, LoRA,
video masks, warm-start strength, first/last frame anchors, multi-frame continuation, or multiple
source videos. Motion-changing prompts can be used on V2V, but MLX-Gen does not claim the full
planner-driven MV2V workflow.

Internally, the Bernini runtime also accepts the official upstream `task_type` values `t2i`,
`i2i`, `t2v`, `r2v`, `rv2v`, `v2v`, `mv2v`, and `ads2v` so the public cases can be replayed
faithfully. Those typed rows are part of the parity harness; the stable user-facing guide here
remains focused on the documented video workflows above.

## Official Public 1.3B Examples

Committed proof for every upstream public example row lives in the
[official public parity bundle](assets/validation/bernini-r-1.3b-2026-08-11/README.md). Each case
directory contains the **full upstream prompt**, input/official/mlx contact sheets, reproduce
command, generated artifact, and metadata when available. Row status follows the
[official example parity matrix](assets/validation/bernini-r-1.3b-2026-08-04/official_example_parity_matrix.md).

| Row | Upstream case | Status | Proof (prompt + sheets) |
| --- | --- | --- | --- |
| `t2i` | `assets/testcases/t2i/t2i.json` | Accepted | [t2i](assets/validation/bernini-r-1.3b-2026-08-11/t2i/README.md) |
| `i2i` | `assets/testcases/i2i/i2i.json` | Accepted (minor wall-texture caveat) | [i2i](assets/validation/bernini-r-1.3b-2026-08-11/i2i/README.md) |
| `t2v` | `assets/testcases/t2v/t2v.json` | Accepted | [t2v](assets/validation/bernini-r-1.3b-2026-08-11/t2v/README.md) |
| `r2v` | `assets/testcases/r2v/r2v.json` | Accepted | [r2v](assets/validation/bernini-r-1.3b-2026-08-11/r2v/README.md) |
| `rv2v_case1` | `assets/testcases/rv2v/rv2v_case1.json` | Accepted (shirt-closure caveat) | [rv2v_case1](assets/validation/bernini-r-1.3b-2026-08-11/rv2v_case1/README.md) |
| `v2v_case1` | `assets/testcases/v2v/v2v_case1.json` | Accepted | [v2v_case1](assets/validation/bernini-r-1.3b-2026-08-11/v2v_case1/README.md) |
| `mv2v` | `assets/testcases/v2v/v2v_case2.json` | Accepted (three named caveats) | [mv2v](assets/validation/bernini-r-1.3b-2026-08-11/mv2v/README.md) |
| `ads2v` | `assets/testcases/rv2v/rv2v_case2.json` | Accepted at mid profile | [ads2v](assets/validation/bernini-r-1.3b-2026-08-11/ads2v/README.md) |

Oracle-dispositioned rows (official recipe fails at 1.3B in both implementations; tuned recipes
recover substantially):

| Row | Recipe | Status | Proof (prompt + sheets) |
| --- | --- | --- | --- |
| `r2v_case2` | Official (`r2v_case2.json`) | Partial at official recipe | [r2v_case2_official](assets/validation/bernini-r-1.3b-2026-08-11/r2v_case2_official/README.md) |
| `r2v_case2` | Tuned (`--reference-guidance 6.0`, seed 43) | Substantial recovery | [r2v_case2_tuned](assets/validation/bernini-r-1.3b-2026-08-11/r2v_case2_tuned/README.md) |
| `v2v_case3` | Official (`v2v_case3.json`) | Official recipe fails at 1.3B | [v2v_case3_official](assets/validation/bernini-r-1.3b-2026-08-11/v2v_case3_official/README.md) |
| `v2v_case3` | Recovery (`mv2v` task prefix, guidance 5.0) | Quadruped dog outcome | [v2v_case3_mv2vprefix](assets/validation/bernini-r-1.3b-2026-08-11/v2v_case3_mv2vprefix/README.md) |

The `r2v_case2` upstream prompt is 571+ UMT5 tokens and truncates to 512 on both engines. The
`ads2v` accepted proof uses the mid profile (`480` condition cap, `61` frames, `24` fps) because
the official `1280x672/121f` recipe is computationally intractable on the validation host for
either implementation.

## Download And Capacity

Download the two pinned, factored sources explicitly:

```sh
mlxgen download --model bernini-r-1.3b
```

This alias selects only `ByteDance/Bernini-R-1.3B-Diffusers`. The main
`ByteDance/Bernini-R` A14B repository and similarly named third-party repositories intentionally
fail closed; MLX-Gen never rewrites them to the 1.3B renderer.

The command downloads all components — tokenizer, the FP32 UMT5 text encoder (loaded and run in
BF16), VAE, and the renderer transformer — from the pinned
`ByteDance/Bernini-R-1.3B-Diffusers` revision. Using the Bernini repository's own text encoder
(not the byte-different Wan-VACE copy) is required for exact official-example parity; this was
verified at the tensor level against the official implementation on 2026-08-12. A completely
cold download is approximately 16.36 GiB of selected files plus 2 GiB of free-space headroom in the
preflight. Existing complete pinned sources are not counted again.

On the 128 GB validation host, the bounded low-RAM profiles peaked at 9.45 GB whole-process
physical footprint. A separate 33-frame, eight-reference 848-condition structural probe peaked at
8.17 GB, so those tested shapes fit inside an 18 GB memory envelope. This is not a direct 18 GB
host measurement. Capacity is separate from the official-example acceptance status above.

## Generate From References

The following commands are diagnostic bounded-profile examples. For official full-profile prompts
and settings, use the [official public parity bundle](assets/validation/bernini-r-1.3b-2026-08-11/README.md).

Reference order is semantic. The prompt names the first image `image0`, the next `image1`, and so
on. Keep each reference focused on the subject or property it is meant to contribute.

```sh
mlxgen generate \
  --model bernini-r-1.3b \
  --reference-image subject.png \
  --reference-image garment.png \
  --prompt "Place the subject from image0 in a studio, wearing the garment from image1, slowly turning toward the camera" \
  --width 320 --height 192 --frames 17 --fps 16 \
  --steps 20 --max-condition-size 256 \
  --seed 42 --low-ram --metadata \
  --output referenced.mp4
```

The renderer supports at most eight references. If the packed conditioning stream has more than
five source segments, source IDs are interpolated across the trained `[1, 5]` range instead of
extrapolating beyond it. The source video counts as one segment in RV2V, so interpolation starts
at five references plus the video; R2V starts at six references.

## Edit A Video With A Reference

```sh
mlxgen generate \
  --model bernini-r-1.3b \
  --video source.mp4 \
  --reference-image garment.jpg \
  --prompt "Replace the outer shirt with the garment from image0 while preserving the person, framing, background, and motion" \
  --width 176 --height 320 --frames 17 --fps 16 \
  --steps 20 --max-condition-size 320 \
  --seed 42 --low-ram --metadata \
  --output garment-edit.mp4
```

Bernini video input always uses `source-aspect` canvas resolution and resize-only conditioning.
Matching the official renderer exactly, the source video's long edge is capped at
`--max-condition-size` (never upscaled, snapped to multiples of 16) and that resolved size is the
output canvas; `--width`/`--height` do not override a video-driven canvas. For source-video modes,
`--frames` is a maximum:
the renderer samples at the requested fps, clamps to the source duration, and returns a `4n+1`
count no greater than the request. Read the returned `num_frames` or metadata `output_frames` for
the actual duration. Crop, pad, exact-resize, `--video-strength`, and `--video-mask-path` are
rejected.

## Prompt-Guided Video Editing

Omit references for source-only V2V:

```sh
mlxgen generate \
  --model bernini-r-1.3b \
  --video source.mp4 \
  --prompt "Add a realistic snowman beside the path while keeping the dog, camera, and winter scene unchanged" \
  --width 320 --height 176 --frames 17 --fps 16 \
  --steps 20 --max-condition-size 320 --seed 42 --low-ram --metadata \
  --output edited.mp4
```

## Defaults And Guidance

The official quality defaults are 848x480, 81 frames, 16 fps, 40 steps, UniPC, and flow shift 5;
they are not the bounded proof profile or an 18 GB-host recommendation. Reference/source inputs
default to a maximum condition side of 848 pixels. The accepted range is 16-1280 in multiples of
16. Prompts have a hard 512-token UMT5 budget and warn when truncated.

| Mode | Active guidance defaults | Inactive options |
| --- | --- | --- |
| R2V | text `4.0`, reference `4.5`, APG eta `0.5`, norm `50`, momentum `0` | source guidance |
| RV2V | text `4.0`, reference `4.5`, source `1.25` | APG controls |
| V2V | text `4.0`, APG eta `0.5`, norm `50`, momentum `0` | reference/source guidance |

Use `--reference-guidance`, `--source-guidance`, `--apg-eta`, `--apg-norm-threshold`, and
`--apg-momentum` only where the table marks them active. Metadata records both active and inactive
parameters, the selected guidance mode, reference order, condition shapes, source IDs, factored
component revisions, prompt truncation, and video health.

## Precision And Quantization

Bernini is currently **BF16-only**. Omit `--quantize`.

The generic Wan q4 policy caused a catastrophic packed-transformer divergence and produced
overexposed latent-like video. Nominal Wan q8 quantized zero Bernini transformer linear layers and
produced a byte-identical BF16 MP4 while reporting q8. MLX-Gen rejects every Bernini quantization
value until a selective low-bit policy passes both numeric and model-backed visual gates.

## Python API

```python
from pathlib import Path

from mflux.models.wan.variants import BerniniRenderer

renderer = BerniniRenderer()
video = renderer.generate_video(
    seed=42,
    prompt="Bring the subject from image0 to life in a fixed medium shot",
    reference_image_paths=[Path("subject.png")],
    width=320,
    height=192,
    num_frames=17,
    fps=16,
    num_inference_steps=20,
    max_condition_size=256,
    clear_cache_each_step=True,
    clear_cache_each_transformer_block=False,
)
video.save("referenced.mp4", export_json_metadata=True)
```

The renderer stays reusable in this form. A one-shot bounded-memory host may pass
`release_denoisers_before_decode=True`, but that releases the renderer before VAE decode; construct
a new `BerniniRenderer` before generating another clip. The CLI enables this destructive release
only for a single-seed low-RAM invocation.

The CLI remains the recommended integration surface for isolated desktop/application jobs because
it supplies final MP4 health validation, failure manifests, output naming, and process-level memory
isolation.

## Proof And Known Limits

The implementation passes scheduler/source-ID tests, APG edge cases, focused FP32/runtime-BF16
transformer comparisons, VAE encode/decode comparisons, and framework regression tests.

### Current official-example evidence

The committed
[official public parity bundle](assets/validation/bernini-r-1.3b-2026-08-11/README.md) holds every
accepted and dispositioned row with full prompts, contact sheets, and generated artifacts. Use the
[official example parity matrix](assets/validation/bernini-r-1.3b-2026-08-04/official_example_parity_matrix.md)
for the row-by-row status, caveats, and oracle methodology.

### Historical schema-v3 release bundle (FAIL)

The older
[schema-v3 validation bundle](assets/validation/bernini-r-1.3b-2026-08-04/README.md) from
2026-08-04 remains historically accurate but superseded for official-example status. It used a
bounded 17-frame/20-step profile and failed visual quality on every required row:

- [Summary contact sheet](assets/validation/bernini-r-1.3b-2026-08-04/bundle/output_summary_contact_sheet.png)
- [Role-control contact sheet](assets/validation/bernini-r-1.3b-2026-08-04/bundle/role_control_contact_sheet.png)
- [Eight-reference MP4](assets/validation/bernini-r-1.3b-2026-08-04/bundle/cases/run_1/r2v_eight_reference/r2v_eight_reference_17f.mp4)
- [Reference-guided edit MP4](assets/validation/bernini-r-1.3b-2026-08-04/bundle/cases/run_1/rv2v_garment/rv2v_garment_17f.mp4)
- [Prompt-guided edit MP4](assets/validation/bernini-r-1.3b-2026-08-04/bundle/cases/run_1/v2v_snowman/v2v_snowman_17f.mp4)
- [Quantization comparison](assets/validation/bernini-r-1.3b-2026-08-04/bundle/diagnostics/quantization_comparison.png)

That bundle found nearly static motion, missed reference properties, cadence-aligned
discontinuities, and severe corruption across frames 13-16 in the garment and snowman cases.
Controlled diagnostics did not rescue the claim: 40 steps retained the 17-frame snowman corruption;
33-frame garment and snowman clips avoided terminal collapse but remained semantically incomplete;
and the exact-upstream-prompt 33-frame eight-reference run collapsed from about frame 13 after
571-token truncation.

The failed experimental route rows are discoverable through the framework validation registry:

```sh
mlxgen validation --model bernini-r-1.3b
```

That command returns `bernini_r_1_3b_2026_08_04` with overall `FAIL`, the exact BF16 package
identity, route-scoped failure notes, upstream input references, and direct paths to the three
model-backed route MP4s from the historical schema-v3 bundle.

The original committed media profile uses 17 frames, 20 steps, reduced canvases, and reduced
condition sizes. Later official-public-case reruns at full profile supersede it for the accepted
rows listed in the parity matrix and parity bundle above.

Controlled diagnostics on the historical bundle did not rescue the claim:

- 40 steps retained the 17-frame snowman corruption;
- 33-frame garment and snowman clips avoided the same terminal collapse but remained semantically
  incomplete and showed periodic four-frame cadence seams;
- a 33-frame run using the exact 2,461-character upstream eight-reference prompt recorded 571
  UMT5 tokens, warned that the last 59 were truncated, and then progressively collapsed from about
  frame 13 into severe geometry and mosaic corruption. Its latent-boundary transition magnitude
  was 2.06 times the non-boundary mean;
- the upstream comparison clips do not attest their producing checkpoint or inference recipe.
  They are qualitative targets, not Bernini-R 1.3B parity baselines.

The remaining promotion work: close the recipe-parity oracle and ablation for the `v2v_case3`
mv2v-prefix recipe, then decide promotion policy for rows whose official recipes are
oracle-proven 1.3B failures but whose documented tuned recipes succeed (`v2v_case3`,
`r2v_case2`). The full `1280x672/121f` ads2v recipe and full-profile official-code oracle
runs remain unverifiable on this host and are recorded as scope limits. Until the policy
decision, keep Bernini experimental and fail closed in release validation.
