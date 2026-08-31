# Reframe And Outpaint

MLX-Gen exposes two single-image canvas expansion workflows through `mlxgen generate`:

- `--reframe-padding` asks an edit model to generate a wider view from the source image. The model
  can redraw the source while changing the crop, viewpoint, or visible subject boundary.
- `--outpaint-padding` expands one source image into a larger canvas and then uses the selected
  edit backend to fill only the added border area as faithfully as that backend allows.

Both options use CSS-style padding in `top,right,bottom,left` order. Percentages are relative to
the source image size. For example, `5%,80%,5%,60%` adds a small top/bottom border, more space to
the right, and a large extension to the left.

`--reframe-padding` is always a generative edit workflow. `--outpaint-padding` is backend-specific:
Qwen Image Edit uses generative canvas expansion with adaptive source restoration, while FLUX.2
Klein routes strict outpaint through base Klein models with source-locked denoising and a narrow
transition band inside the source crop. Neither route is a native masked fill/inpaint pipeline, so
review the output visually.

## The Conditioning Canvas

Outpaint pastes your source onto a larger canvas and asks the model to complete the added area, so
what fills that area *before* denoising decides what you get back. On FLUX.2 Klein base you choose
that with `--outpaint-fill`:

| Mode | What it paints | Use it when |
| --- | --- | --- |
| `auto` (default) | Picks one of the modes below from the padding depth and the loaded adapter, and prints which and why. | You want a sensible canvas without thinking about it. |
| `edge` | Stretches the source border strip outward. | You are continuing an existing texture across a border within the edge-fill reach. |
| `neutral` | A flat per-side border color sampled from the source. | You are adding a lot of space and want the model to invent new subject matter. |
| `solid` | One flat color, from `--outpaint-fill-color`. | You need an exact canvas color, for example an adapter trained on one. |
| `blur` | A blurred, scaled copy of the source. | You want a soft background suggestion rather than a blank one. |

Edge fill continues a texture; it does not invent one. It works by stretching a source border strip
across the padded area, so it holds up while the padding stays within what that strip covers — the
**edge-fill reach**, printed for every run. Past the reach the strip is stretched far enough that it
reads as directional streaks rather than continued texture, so `auto` switches to `neutral` there:
a blank canvas gives the model nothing to continue and it generates instead.

`auto` also switches to `solid` green when a green-border outpaint adapter is loaded, because those
adapters are trained to paint into a pure-green canvas.

Every outpaint run prints its resolved canvas:

```
Outpaint: fill=neutral, canvas 928x1536 from source 768x766, padding top=0 right=76 bottom=766 left=76.
Outpaint: --outpaint-fill auto selected neutral because the deepest padding is bottom 766px (100% of
the source height), 2.0x the 384px edge-fill reach; a blank canvas makes the model generate new
subject matter instead of smearing the source border.
```

Applications can read the same contract as JSON without running a job — `outpaint_fill_modes`,
`outpaint_default_fill_mode`, `outpaint_auto_edge_fill_max_stretch`, `outpaint_recommended_lora`,
and the validated envelope are published on the capability record. See
[Edit Capabilities](edit-capabilities.md).

## Choosing Padding

Padding is the single biggest factor in outpaint quality. Two guidelines:

- **Extend in steps rather than in one jump.** Two passes of 50-70% give the model a nearby edge to
  continue from each time, and each pass runs on a smaller canvas.
- **Watch the canvas size.** Outpaint conditioning tokens scale with the canvas, and attention cost
  grows faster than area. A 1.4 MP canvas costs substantially more per step than a 0.3 MP one.

For revealing a subject the source crop never showed — a full body from a head-and-shoulders
portrait, for example — `neutral` fill with a descriptive prompt is the route that generates new
subject matter. A blank canvas gives the model room to invent but little guidance on where the
subject continues, so at very deep padding the new content can sit away from the source edge. The
outpaint adapter in [LoRA](lora.md) is the recommended configuration for this case and holds the
continuation to the source; extending in two moderate passes helps for the same reason.

## Supported Models

The historical mixed validation profile is `reframe_outpaint_2026_06_08`. It uses one cropped
starship source image. Treat the distilled FLUX.2 outpaint rows in that profile as historical
artifacts only; they are no longer the current FLUX.2 outpaint contract. Current source-model
FLUX.2 Klein base proof is published separately as `flux2_klein_base_starship_2026_06_10`.

| Family | Reframe | Outpaint | Notes |
| --- | --- | --- | --- |
| Qwen Image Edit / 2509 / 2511 | current | current | Published `reframe_outpaint_2026_06_08` profile remains representative for the base route. Exact 2511 q8 LoRA-backed reframe and outpaint rows are published separately in [LoRA](lora.md). |
| FLUX.2 Klein 4B / 9B distilled | current | historical only | Reframe remains supported. Historical outpaint rows are stale and are no longer exposed as strict outpaint. |
| FLUX.2 Klein Base 4B / 9B | not exposed | current | Strict FLUX.2 outpaint now requires a base Klein model. Source-model contact sheets are published, and the exact `AbstractFramework/flux.2-klein-base-4b-8bit` q8 LoRA-backed outpaint row is published separately in [LoRA](lora.md). |

These options are intentionally not exposed for base Qwen Image, Qwen Image 2512, ERNIE Image
Turbo, Z-Image, FIBO, Bonsai, Wan, or SeedVR2. Those families are text generation, latent I2I,
video, upscale/restoration routes, or do not yet have a validated edit-reference canvas-expansion
profile.

Check support before running:

```sh
mlxgen capabilities --model AbstractFramework/qwen-image-edit-2511-8bit
```

Inspect the June 8 mixed-profile validation records for a package:

```sh
mlxgen validation \
  --profile reframe_outpaint_2026_06_08 \
  --model AbstractFramework/qwen-image-edit-2511-8bit
```

Inspect the current base-source starship validation records:

```sh
mlxgen validation \
  --profile flux2_klein_base_starship_2026_06_10 \
  --model black-forest-labs/FLUX.2-klein-base-9B
```

## Reframe Example

Use reframe when you want a model to create a wider view and you accept that the source may be
redrawn:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-8bit \
  --image input.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the full subject and extend the background naturally." \
  --steps 16 \
  --seed 42 \
  --output reframed.png
```

## Outpaint Example

Use outpaint when you want MLX-Gen to expand one source image while keeping the original crop as
stable as the backend allows:

```sh
mlxgen generate \
  --model black-forest-labs/FLUX.2-klein-base-9B \
  --image input.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close crop into a wider realistic shot. Complete the missing subject and background outside the original frame." \
  --steps 20 \
  --guidance 4 \
  --seed 42 \
  --output outpaint.png
```

To add a lot of space on one side — extending a portrait downward to reveal more of the subject —
pass the padding on that side and let `auto` pick the blank canvas, or name it explicitly:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-base-4b-8bit \
  --image portrait.png \
  --outpaint-padding "0%,10%,100%,10%" \
  --outpaint-fill neutral \
  --prompt "Extend this portrait downward to reveal the lower part of the body: the same subject in the same clothing, same lighting, same background." \
  --steps 20 \
  --guidance 4 \
  --seed 1234 \
  --output extended.png
```

`--outpaint-padding` computes the output size from the source and the padding, so do not pass
`--width`, `--height`, or `--canvas-policy` with it.

For Qwen Image Edit variants, MLX-Gen may still apply adaptive source restoration after generation.
For current FLUX.2 Klein base outpaint, MLX-Gen relies on source-locked denoising with an interior
transition band instead of pasting the original crop back over the final image.

## Validation Assets

The current proof set uses this source image:

![Cropped starship source](assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png)

The outpaint helper creates this wider conditioning canvas and source-window mask:

![Wide outpaint canvas](assets/validation/reframe-outpaint-2026-06-08/source-b-outpaint-canvas-wide.png)

In the mask image, black marks the original source window and white marks the generated border area.

![Wide outpaint source mask](assets/validation/reframe-outpaint-2026-06-08/source-b-outpaint-mask-wide.png)

The summary sheet shows the historical 2026-06-08 source/q8/q4 rows:

![Reframe and outpaint source/q8/q4 summary](assets/validation/reframe-outpaint-2026-06-08/reframe-outpaint-base-q8-q4-summary.jpg)

Per-family contact sheets:

- [Qwen Image Edit](assets/validation/reframe-outpaint-2026-06-08/qwen-image-edit-reframe-outpaint-matrix.jpg)
- [Qwen Image Edit 2509](assets/validation/reframe-outpaint-2026-06-08/qwen-image-edit-2509-reframe-outpaint-matrix.jpg)
- [Qwen Image Edit 2511](assets/validation/reframe-outpaint-2026-06-08/qwen-image-edit-2511-reframe-outpaint-matrix.jpg)
- [FLUX.2 Klein 4B](assets/validation/reframe-outpaint-2026-06-08/flux2-klein-4b-reframe-outpaint-matrix.jpg) - historical distilled reframe/outpaint matrix
- [FLUX.2 Klein 9B](assets/validation/reframe-outpaint-2026-06-08/flux2-klein-9b-reframe-outpaint-matrix.jpg) - historical distilled reframe/outpaint matrix

Current source-model FLUX.2 Klein base proof:

- [Base 4B/9B edit and strict-outpaint matrix](assets/validation/flux2-klein-base-starship-2026-06-10/flux2-klein-base-starship-edit-matrix.jpg)
- [Base 4B/9B strict-outpaint seam review](assets/validation/flux2-klein-base-starship-2026-06-10/flux2-klein-base-starship-outpaint-seams.jpg)
- [Base 4B/9B text-to-image smoke panel](assets/validation/flux2-klein-base-starship-2026-06-10/flux2-klein-base-starship-t2i-panel.jpg)

The exact commands and validation manifest are published with the assets:

- [Command log](assets/validation/reframe-outpaint-2026-06-08/reframe-outpaint-command-log.md)
- [Validation manifest](assets/validation/reframe-outpaint-2026-06-08/reframe-outpaint-validation-manifest.json)
- [Base starship command log](assets/validation/flux2-klein-base-starship-2026-06-10/flux2-klein-base-starship-command-log.md)
- [Base starship validation manifest](assets/validation/flux2-klein-base-starship-2026-06-10/flux2-klein-base-starship-validation-manifest.json)
