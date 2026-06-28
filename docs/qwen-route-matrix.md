# Qwen Route Matrix

This page is the current route truth for Qwen Image and Qwen Image Edit in MLX-Gen. It maps the
public `mlxgen` surface to the upstream Diffusers Qwen pipeline families and points at the exact
proof assets that already exist.

Use this page for three questions:

- which Qwen route does MLX-Gen actually expose;
- what user inputs select that route;
- where the current public proof for that route lives.

For exact per-model option support, use:

```sh
mlxgen capabilities --model AbstractFramework/qwen-image-edit-2511-8bit
```

For exact release evidence on one row, use:

```sh
mlxgen validation --model AbstractFramework/qwen-image-edit-2511-8bit
```

## Current Route Map

| Capability id | Upstream parity | Typical handles | Public inputs | Current proof surface |
| --- | --- | --- | --- | --- |
| `qwen.text` | `QwenImagePipeline` | `Qwen/Qwen-Image`, `AbstractFramework/qwen-image-8bit`, `AbstractFramework/qwen-image-2512-8bit` | `--prompt` | Route is shipped. Current exact LoRA-backed proof rows exist for `AbstractFramework/qwen-image-8bit` and `AbstractFramework/qwen-image-2512-8bit` in [LoRA](lora.md). |
| `qwen.latent` | `QwenImageImg2ImgPipeline` | base Qwen Image handles | `--image + --image-strength + --prompt` | Route is shipped. An exact LoRA-backed public proof now exists for `AbstractFramework/qwen-image-8bit`; see [LoRA](lora.md). |
| `qwen.edit` | `QwenImageEditPipeline` | `Qwen/Qwen-Image-Edit`, `Qwen/Qwen-Image-Edit-2509`, `Qwen/Qwen-Image-Edit-2511`, prepared q8/q4 variants | `--image + --prompt` | [Image edit capabilities](edit-capabilities.md) covers the source/q8/q4 edit matrices. |
| `qwen.inpaint` | `QwenImageEditInpaintPipeline` | edit-model rows, with the current exact proof on `AbstractFramework/qwen-image-edit-2511-8bit` | `--image + --mask-path + --prompt` | [Qwen localized editing](qwen-localized-editing.md) links the accepted masked-edit proof. |
| `qwen.multi-reference` | `QwenImageEditPlusPipeline` behavior | `Qwen/Qwen-Image-Edit-2509`, `Qwen/Qwen-Image-Edit-2511`, prepared q8/q4 variants | repeated `--image` + `--prompt` | [Image edit capabilities](edit-capabilities.md) links the accepted 2509 and 2511 multi-reference matrices, and [LoRA](lora.md) covers the exact 2511 q8 multi-angle proof row. |
| `qwen.reframe` | MLX-Gen canvas expansion on the Qwen edit route | original/2509/2511 edit families and prepared variants | `--image + --reframe-padding + --prompt` | [Reframe and outpaint](reframe-outpaint.md) links the accepted Qwen reframe matrices, and [LoRA](lora.md) covers the exact 2511 q8 multi-angle proof row. |
| `qwen.outpaint` | MLX-Gen canvas expansion on the Qwen edit route | original/2509/2511 edit families and prepared variants | `--image + --outpaint-padding + --prompt` | [Reframe and outpaint](reframe-outpaint.md) links the accepted Qwen outpaint matrices, and [LoRA](lora.md) covers the exact 2511 q8 multi-angle proof row. |
| `qwen.control` | `QwenImageControlNetPipeline` | current exact public row: `AbstractFramework/qwen-image-8bit` | `--controlnet-image-path + --prompt` | [Qwen localized editing](qwen-localized-editing.md) links the accepted structured-control proof. |
| `qwen.control-inpaint` | `QwenImageControlNetInpaintPipeline` | current exact public row: `AbstractFramework/qwen-image-8bit` | `--image + --mask-path + --prompt` | [Qwen localized editing](qwen-localized-editing.md) links the accepted control-inpaint proof. |

## Important Public-Boundary Choices

MLX-Gen intentionally does not expose every upstream Qwen pipeline name as a separate public route
id.

The current choices are:

- `QwenImageEditPlusPipeline` behavior is exposed through `qwen.multi-reference`, not through a
  separate public `qwen.edit-plus` id;
- base-Qwen masked work is exposed through `qwen.control-inpaint`, not through a separate public
  native-base inpaint route;
- `QwenImageLayeredPipeline` is not a shipped public MLX-Gen route today.

That keeps the user-facing contract smaller while preserving honest route identity through
`mlxgen capabilities`.

## Which Qwen Route To Choose

| Need | Route |
| --- | --- |
| Generate from text only | `qwen.text` |
| Restyle one existing image with explicit `--image-strength` | `qwen.latent` |
| Edit one source image with an instruction | `qwen.edit` |
| Edit only a masked part of one source image on the edit checkpoint | `qwen.inpaint` |
| Combine two or more reference images | `qwen.multi-reference` |
| Generate from text while fixing layout from a control image | `qwen.control` |
| Edit a masked source region on base Qwen with the stricter inpaint sidecar | `qwen.control-inpaint` |
| Expand the canvas outward | `qwen.reframe` or `qwen.outpaint`, depending on whether you want zoom-out or canvas extension |

## Related Docs

- [API and CLI](api.md)
- [Image edit modes](image-edit-modes.md)
- [Image edit capabilities](edit-capabilities.md)
- [Qwen localized editing](qwen-localized-editing.md)
- [LoRA](lora.md)
