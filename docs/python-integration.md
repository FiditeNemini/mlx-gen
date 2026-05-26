# Python Integration

MLX-Gen can be embedded directly in Python. The current runtime still exposes most model classes through the original `mflux` package layout, with `mlxgen` available as the package identity for new applications.

## Cache-Only Runtime

Python callers should download or prepare models before constructing model objects. Runtime constructors and generation calls do not download missing artifacts. See [Model Management](model-management.md) for the CLI setup commands.

```python
from mflux.models.common.download_policy import DownloadRequiredError
from mlxgen.models.z_image import ZImageTurbo

try:
    model = ZImageTurbo(quantize=8)
except DownloadRequiredError as exc:
    print(exc.download_command)
    raise
```

For user-facing applications, show the exception message or the `download_command`/`prepare_command` fields and stop the workflow.

## AbstractVision

MLX-Gen is intended to be the Apple Silicon / MLX dependency for AbstractVision while AbstractVision remains a cross-platform orchestration package.

That split means:

- AbstractVision owns provider-neutral request objects, artifact storage, capability checks, and AbstractCore plugin integration.
- MLX-Gen owns MLX model loading, model-family behavior, local quantized formats, and runtime compatibility fixes.
- MLX-Gen should fail early when required local artifacts are missing so AbstractVision can surface a clear remediation message instead of starting a network transfer.

The current integration path is still model-specific Python classes. A future higher-level facade may expose explicit prepared/loaded/warmed model states, but current docs only describe the APIs that exist now.

## Progress And Monitoring

Existing model callbacks remain available through the mflux runtime internals. Wan video generation also exposes a direct `progress_callback` hook for applications that need frame-oriented progress.

```python
from mflux.models.wan.variants import Wan2_2_TI2V, WanProgressEvent


def on_progress(event: WanProgressEvent) -> None:
    print(
        f"{event.phase}: frame {event.frame}/{event.total_frames}, "
        f"step {event.step}/{event.total_steps}, {event.progress:.0%}"
    )


model = Wan2_2_TI2V(model_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
video = model.generate_video(
    seed=321,
    prompt="A slow cinematic shot of a glass sphere floating above teal water",
    width=1280,
    height=704,
    num_frames=121,
    num_inference_steps=50,
    fps=24,
    progress_callback=on_progress,
)
video.save("video.mp4")
```

Wan progress events use output-frame units so callers can display `frame / total_frames`, while also receiving the active denoising `step / total_steps`. Other backends still use their existing model-specific callback paths.

## Threading

MLX model instances should be treated as stateful runtime objects. Applications that multiplex user requests should serialize access to a loaded model instance unless they have tested a narrower concurrency model for that specific backend.
