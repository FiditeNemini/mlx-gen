# Model Management

MLX-Gen generation is cache-only by default. It will not download model weights, tokenizers, LoRAs, or Depth Pro weights during generation or during ordinary Python model construction.

This policy keeps CLI jobs and embedded application workflows predictable: a generation request either finds the required files locally or fails with a `DownloadRequiredError` that includes the command to run.

## Download A Hugging Face Snapshot

Use `mlxgen download` to populate the local Hugging Face cache:

```sh
HF_HUB_ENABLE_HF_TRANSFER=1 mlxgen download --model Qwen/Qwen-Image
```

Aliases are supported when MLX-Gen knows the model:

```sh
HF_HUB_ENABLE_HF_TRANSFER=1 mlxgen download --model z-image-turbo
```

For LoRA repositories, download the repository explicitly before passing it to generation:

```sh
HF_HUB_ENABLE_HF_TRANSFER=1 mlxgen download --model RiverZ/normal-lora --all-files
```

## Prepare A Local MLX-Gen Folder

Use `mlxgen prepare` when you want a reusable local folder, usually with quantized MLX-Gen weights:

```sh
HF_HUB_ENABLE_HF_TRANSFER=1 mlxgen prepare \
  --model Qwen/Qwen-Image \
  --path ./models/qwen-image-8bit \
  -q 8
```

Then generate from the local folder:

```sh
mlxgen generate \
  --model ./models/qwen-image-8bit \
  --family qwen \
  --prompt "A product photo of a ceramic teapot" \
  --output image.png
```

`mflux-save` remains available for compatibility and uses the same explicit download permission as `mlxgen prepare`.

## Depth Pro

Depth workflows use Apple Depth Pro weights from a direct URL rather than a Hugging Face repository. Download them explicitly:

```sh
mlxgen download --model depth-pro
```

After that, `mflux-save-depth` and depth generation can run without starting a network transfer.

## Runtime Failure Contract

When files are missing, MLX-Gen raises `DownloadRequiredError`. The exception is also a `FileNotFoundError` for compatibility with existing callers. It exposes:

- `repo_id`
- `artifact`
- `download_command`
- `prepare_command` when a local prepared folder is applicable

The human-readable message is designed for non-expert users and includes the exact command to run.
