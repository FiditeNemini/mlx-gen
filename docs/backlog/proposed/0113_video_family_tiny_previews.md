# Proposed: tiny previews for the Wan, Qwen, and FIBO latent spaces (TAEHV family)

## Metadata

- Created: 2026-08-14
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: none directly.
- ADR impact: none; preview-only surface, never on the save path.

## Context

Tiny-decoder previews shipped for the FLUX.1 latent space (`taef1`, covering flux and Z-Image)
and the FLUX.2 latent space (`taef2`, covering FLUX.2 Klein, ERNIE-Image, and Bonsai). The
remaining families need the **TAEHV** architecture, which is a different codebase from TAESD:
NTCHW layout, causal-in-time `MemBlock`s that consume the previous timestep's activation, and
`TGrow`/`TPool` temporal resampling convolutions.

Upstream VAE lineage was established by direct safetensors comparison (2026-08-14), so the
mapping is known and does not need re-deriving:

| Our family | latent channels | checkpoint |
| --- | --- | --- |
| qwen (Image, Edit, 2509, 2511) | 16 | `taew2_1` |
| wan 2.2 A14B (T2V/I2V), wan 2.1 VACE, bernini | 16 | `taew2_1` |
| wan 2.2 TI2V-5B | 48 | `taew2_2` |
| fibo (all variants) | 48 | `taew2_2` (unverified: repository is gated) |
| seedvr2 | 16 | none published |

Qwen-Image ships Wan 2.1's encoder verbatim with a retrained decoder, so `taew2_1` previews
approximate Wan's rendering of a Qwen latent. Expect a small systematic offset beyond ordinary
tiny-decoder error, and validate it empirically before enabling the mapping.

## Why it matters

These are the slowest routes we have. A Wan video run is minutes long, which makes it the single
highest-value place to see progress early — the argument that motivated previews in the first
place. Video previews also need the temporal handling that the image tiny autoencoders correctly
refuse to fake (they raise on multi-frame latents today).

## Proposed work

- Port `MemBlock`, `TGrow`, `TPool`, and the TAEHV decoder graph to MLX alongside the existing
  `TinyAutoencoder`, including `pixel_shuffle` postprocessing for the patch-2 checkpoints and the
  `frames_to_trim` leading-frame drop.
- Load the raw (non-diffusers) key layout these checkpoints publish, and their float16 storage.
- Extend the preview registry with the `wan.2.1`, `wan.2.2-ti2v`, and `qwen-image` latent spaces,
  each enabled only after an empirical check against that family's own VAE.
- Validate per family with the existing harness (`untracked/taesd_consistency/`): decode real
  in-flight latents with both decoders and report agreement, plus a negative control.

## Risks

- Single-frame image use of a video decoder is supported upstream but produces `t_upscale`
  frames that must be trimmed; getting the trim wrong silently shows the wrong frame.
- The Qwen decoder mismatch above means a preview can be self-consistent yet systematically off
  the final image; document it rather than tuning it away.
