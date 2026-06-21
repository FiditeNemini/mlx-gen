import gc
import time
from dataclasses import asdict, replace
from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.seedvr2.latent_creator.seedvr2_latent_creator import SeedVR2LatentCreator
from mflux.models.seedvr2.model.seedvr2_text_encoder.text_embeddings import SeedVR2TextEmbeddings
from mflux.models.seedvr2.model.seedvr2_transformer.transformer import SeedVR2Transformer
from mflux.models.seedvr2.model.seedvr2_vae.vae import SeedVR2VAE
from mflux.models.seedvr2.seedvr2_initializer import SeedVR2Initializer
from mflux.models.seedvr2.variants.upscale.seedvr2_util import SeedVR2Util
from mflux.models.seedvr2.weights.seedvr2_weight_definition import SeedVR2WeightDefinition
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.generated_video import GeneratedVideo
from mflux.utils.image_util import ImageUtil
from mflux.utils.metadata_reader import MetadataReader
from mflux.utils.scale_factor import ScaleFactor
from mflux.utils.video_health import VideoHealth
from mflux.utils.video_util import AudioCopyResult, VideoStreamWriter, VideoUtil


class SeedVR2(nn.Module):
    vae: SeedVR2VAE
    transformer: SeedVR2Transformer

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig = ModelConfig.seedvr2_3b(),
    ):
        super().__init__()
        SeedVR2Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config,
        )

    def generate_image(
        self,
        seed: int,
        image_path: str | Path,
        resolution: int | ScaleFactor,
        softness: float = 0.0,
        color_correction_mode: str = "wavelet",
    ) -> GeneratedImage:
        # 0. Process and scale the input image
        processed_image, true_height, true_width = SeedVR2Util.preprocess_image(
            image_path=image_path,
            resolution=resolution,
            softness=softness,
        )
        tiling_config = self._effective_tiling_config(true_height=true_height, true_width=true_width)

        # 1. Create a new config based on the model type and input parameters
        config = Config(
            width=true_width,
            height=true_height,
            guidance=1.0,
            num_inference_steps=1,
            image_path=image_path,
            scheduler="seedvr2_euler",
            model_config=self.model_config,
            dimension_multiple=2,
        )

        # 2. Create the initial latents and conditioning
        initial_latent = VAEUtil.encode(vae=self.vae, image=processed_image, tiling_config=tiling_config)
        static_condition = SeedVR2LatentCreator.create_condition(encoded_latent=initial_latent)
        latents = SeedVR2LatentCreator.create_noise_latents(seed=seed, height=initial_latent.shape[-2], width=initial_latent.shape[-1])  # fmt: off

        # 3. Get the pre-computed text embeddings
        text_embedding = getattr(self, "text_embedding", None)
        txt_pos = (
            SeedVR2TextEmbeddings.prepare_positive(text_embedding)
            if text_embedding is not None
            else SeedVR2TextEmbeddings.load_positive()
        )

        # 4. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt="", config=config)
        ctx.before_loop(latents)

        for t in config.time_steps:
            model_input = mx.concatenate([latents, static_condition], axis=1)

            # 5.t Predict the noise
            noise = self.transformer(
                txt=txt_pos,
                vid=model_input,
                timestep=config.scheduler.timesteps[t],
            )

            # 6.t Take one denoise step
            latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)

            # 7.t Call subscribers in-loop
            ctx.in_loop(t, latents)

            mx.eval(latents)

        # 8. Call subscribers after loop
        ctx.after_loop(latents)

        # 9. Decode the latents and return the image
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=tiling_config)
        decoded = decoded[:, :, :true_height, :true_width]
        style = processed_image[:, :, :true_height, :true_width]
        decoded = SeedVR2Util.apply_color_correction(decoded, style, mode=color_correction_mode)

        # 10. Read metadata from the original image if available
        init_metadata = MetadataReader.read_all_metadata(image_path) if image_path else None

        generated_image = ImageUtil.to_image(
            seed=seed,
            prompt="",
            config=config,
            quantization=self.bits,
            decoded_latents=decoded,
            generation_time=config.time_steps.format_dict["elapsed"],
            image_path=image_path,
            init_metadata=init_metadata,
            extra_metadata={
                "resolution": str(resolution),
                "softness": round(float(softness), 3),
                "color_correction_mode": color_correction_mode,
                **self._seedvr2_metadata(),
            },
        )
        del decoded
        mx.clear_cache()
        gc.collect()
        return generated_image

    def generate_video(
        self,
        seed: int,
        video_path: str | Path,
        resolution: int | ScaleFactor,
        softness: float = 0.0,
        start_seconds: float = 0.0,
        max_frames: int | None = None,
        color_correction_mode: str = "wavelet",
        restore_metadata: dict | None = None,
    ):
        start_time = time.perf_counter()
        clip_probe = VideoUtil.read_video_clip(
            path=video_path,
            start_seconds=start_seconds,
            max_frames=1,
        )
        if clip_probe.source_frame_count is not None:
            available_frames = max(0, clip_probe.source_frame_count - clip_probe.clip_start_frame)
        elif clip_probe.source_duration_seconds is not None:
            available_frames = max(1, int(round(clip_probe.source_duration_seconds * clip_probe.fps)) - clip_probe.clip_start_frame)
        else:
            raise RuntimeError("SeedVR2 video restore requires a finite source frame count or duration.")
        requested_clip_frames = min(max_frames, available_frames) if max_frames is not None else available_frames
        if requested_clip_frames > 1:
            raise ValueError(
                "SeedVR2.generate_video() is limited to single-frame in-memory use. "
                "Use restore_video_to_path() or `mlxgen upscale --video-path ...` for streamed multi-frame restore."
            )
        self._assert_generate_video_supported(
            frame_count=requested_clip_frames,
            source_width=clip_probe.source_width,
            source_height=clip_probe.source_height,
            resolution=resolution,
        )
        video_clip = VideoUtil.read_video_clip(
            path=video_path,
            start_seconds=start_seconds,
            max_frames=requested_clip_frames,
        )
        decoded, true_height, true_width, padded_input_frames = self._restore_video_frames(
            seed=seed,
            frames=video_clip.frames,
            resolution=resolution,
            softness=softness,
            color_correction_mode=color_correction_mode,
        )
        generation_time = time.perf_counter() - start_time

        generated_video = VideoUtil.to_video(
            decoded_latents=decoded,
            fps=video_clip.fps,
            model_config=self.model_config,
            seed=seed,
            prompt="",
            steps=1,
            guidance=1.0,
            quantization=self.bits,
            generation_time=generation_time,
            task="video-to-video",
            video_path=video_path,
            extra_metadata={
                "resolution": str(resolution),
                "softness": round(float(softness), 3),
                **self._seedvr2_metadata(),
                **(restore_metadata or {}),
                "source_video_width": video_clip.source_width,
                "source_video_height": video_clip.source_height,
                "source_video_fps": round(float(video_clip.fps), 6),
                "source_video_frames": video_clip.source_frame_count,
                "source_video_duration_seconds": (
                    round(video_clip.source_duration_seconds, 3)
                    if video_clip.source_duration_seconds is not None
                    else None
                ),
                "source_clip_start_frame": video_clip.clip_start_frame,
                "source_clip_start_seconds": round(float(start_seconds), 3),
                "source_clip_actual_start_seconds": round(float(video_clip.clip_start_frame / video_clip.fps), 6),
                "source_clip_frames": video_clip.clip_frame_count,
                "padded_input_frames": padded_input_frames,
                "audio_present": video_clip.audio_present,
                "audio_copied": False,
                "audio_copy_mode": None,
                "audio_copy_reason": "in_memory_output",
                "temporal_chunk_size": None,
                "temporal_chunk_overlap": None,
                "color_correction_mode": color_correction_mode,
            },
        )
        del decoded
        mx.clear_cache()
        gc.collect()
        return generated_video

    def restore_video_to_path(
        self,
        *,
        seed: int,
        video_path: str | Path,
        resolution: int | ScaleFactor,
        softness: float = 0.0,
        start_seconds: float = 0.0,
        max_frames: int | None = None,
        output_path: str | Path,
        export_json_metadata: bool = False,
        overwrite: bool = True,
        validate_health: bool = True,
        temporal_chunk_size: int = 49,
        temporal_chunk_overlap: int = 16,
        color_correction_mode: str = "wavelet",
        drop_audio: bool = False,
        restore_metadata: dict | None = None,
        enforce_memory_budget: bool = True,
    ) -> Path:
        start_time = time.perf_counter()
        clip_probe = VideoUtil.read_video_clip(
            path=video_path,
            start_seconds=start_seconds,
            max_frames=1,
        )
        if clip_probe.source_frame_count is not None:
            available_frames = max(0, clip_probe.source_frame_count - clip_probe.clip_start_frame)
        elif clip_probe.source_duration_seconds is not None:
            available_frames = max(1, int(round(clip_probe.source_duration_seconds * clip_probe.fps)) - clip_probe.clip_start_frame)
        else:
            raise RuntimeError("SeedVR2 video restore requires a finite source frame count or duration.")
        requested_clip_frames = min(max_frames, available_frames) if max_frames is not None else available_frames
        actual_clip_start_seconds = clip_probe.clip_start_frame / clip_probe.fps
        chunk_plan = SeedVR2Util.plan_streamed_video_chunks(
            frame_count=requested_clip_frames,
            chunk_size=temporal_chunk_size,
            overlap=temporal_chunk_overlap,
        )
        global_noise = self._build_streamed_video_noise_latents(
            seed=seed,
            requested_clip_frames=requested_clip_frames,
            source_width=clip_probe.source_width,
            source_height=clip_probe.source_height,
            resolution=resolution,
        )

        final_width: int | None = None
        final_height: int | None = None
        writer = None
        file_path: Path | None = None
        audio_copy_result = None
        try:
            chunk_clips = VideoUtil.iter_video_frame_windows(
                video_path,
                start_frame=clip_probe.clip_start_frame,
                windows=[(chunk.input_start_frame, chunk.input_end_frame) for chunk in chunk_plan],
            )
            for chunk_index, chunk_clip in enumerate(chunk_clips):
                try:
                    mx.reset_peak_memory()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
                chunk_info = chunk_plan[chunk_index]
                chunk_frames = list(chunk_clip.frames)
                if len(chunk_frames) < chunk_info.target_input_frame_count:
                    if not chunk_frames:
                        raise ValueError("SeedVR2 streamed video chunk decode returned no frames.")
                    pad_frame = chunk_frames[-1]
                    while len(chunk_frames) < chunk_info.target_input_frame_count:
                        chunk_frames.append(pad_frame.copy())
                noise_slice = self._streamed_video_noise_slice(
                    global_noise=global_noise,
                    chunk_start_frame=chunk_info.input_start_frame,
                    target_input_frame_count=chunk_info.target_input_frame_count,
                )
                decoded, true_height, true_width, _ = self._restore_video_frames(
                    seed=seed,
                    frames=chunk_frames,
                    resolution=resolution,
                    softness=softness,
                    color_correction_mode=color_correction_mode,
                    enforce_memory_budget=enforce_memory_budget,
                    noise_latents=noise_slice,
                )
                final_width = true_width
                final_height = true_height

                body_start = chunk_info.trim_leading_context_frames
                body_end = body_start + chunk_info.output_frame_count
                batch_to_write = []
                frame_arrays_to_write = None
                if body_end > body_start:
                    if isinstance(decoded, list):
                        batch_to_write = decoded[body_start:body_end]
                    else:
                        frame_arrays_to_write = VideoUtil._latents_to_frame_arrays(decoded[:, :, body_start:body_end, :, :])
                if batch_to_write or frame_arrays_to_write is not None:
                    first_width = batch_to_write[0].width if batch_to_write else int(frame_arrays_to_write.shape[2])
                    first_height = batch_to_write[0].height if batch_to_write else int(frame_arrays_to_write.shape[1])
                    if writer is None:
                        writer = VideoStreamWriter(
                            path=output_path,
                            fps=clip_probe.fps,
                            width=first_width,
                            height=first_height,
                            overwrite=overwrite,
                        )
                    if batch_to_write:
                        writer.write_frames(batch_to_write)
                    else:
                        writer.write_frame_arrays(frame_arrays_to_write)

                del chunk_frames
                del batch_to_write
                del frame_arrays_to_write
                del noise_slice
                del decoded
                mx.clear_cache()
                gc.collect()
                self._assert_post_chunk_memory_health(
                    true_height=true_height,
                    true_width=true_width,
                    frame_count=chunk_info.target_input_frame_count,
                    enforce_peak_budget=enforce_memory_budget,
                )

            if writer is None:
                raise ValueError("SeedVR2 video restore did not produce any frames.")
            file_path = writer.close()
            audio_copy_result = AudioCopyResult(
                audio_present=False,
                audio_copied=False,
                copy_mode=None,
                reason="no_source_audio",
            )
            if clip_probe.audio_present:
                if drop_audio:
                    audio_copy_result = AudioCopyResult(
                        audio_present=True,
                        audio_copied=False,
                        copy_mode=None,
                        reason="drop_audio_requested",
                    )
                else:
                    audio_copy_result = VideoUtil.copy_source_audio_to_video(
                        source_video_path=video_path,
                        restored_video_path=file_path,
                        clip_start_seconds=actual_clip_start_seconds,
                        clip_duration_seconds=requested_clip_frames / clip_probe.fps,
                    )
                    if not audio_copy_result.audio_copied:
                        raise RuntimeError(
                            "Source audio was present but MLX-Gen could not preserve it safely "
                            f"({audio_copy_result.reason}). Re-run with drop_audio=True or --drop-audio "
                            "to allow a silent restored MP4 intentionally."
                        )
        except Exception:
            if writer is not None:
                writer.abort()
            if file_path is not None:
                SeedVR2._cleanup_video_artifacts(file_path)
            raise

        try:
            generation_time = time.perf_counter() - start_time
            metadata = GeneratedVideo.build_metadata(
                model_config=self.model_config,
                seed=seed,
                prompt="",
                steps=1,
                guidance=1.0,
                guidance_2=None,
                flow_shift=None,
                solver=None,
                precision=ModelConfig.precision,
                quantization=self.bits,
                generation_time=generation_time,
                height=final_height or 0,
                width=final_width or 0,
                frame_count=requested_clip_frames,
                fps=clip_probe.fps,
                task="video-to-video",
                video_path=video_path,
                extra_metadata={
                    "resolution": str(resolution),
                    "softness": round(float(softness), 3),
                    **self._seedvr2_metadata(),
                    **(restore_metadata or {}),
                    "source_video_width": clip_probe.source_width,
                    "source_video_height": clip_probe.source_height,
                    "source_video_fps": round(float(clip_probe.fps), 6),
                    "source_video_frames": clip_probe.source_frame_count,
                    "source_video_duration_seconds": (
                        round(clip_probe.source_duration_seconds, 3)
                        if clip_probe.source_duration_seconds is not None
                        else None
                    ),
                    "source_clip_start_frame": clip_probe.clip_start_frame,
                    "source_clip_start_seconds": round(float(start_seconds), 3),
                    "source_clip_actual_start_seconds": round(float(actual_clip_start_seconds), 6),
                    "source_clip_frames": requested_clip_frames,
                    "padded_input_frames": SeedVR2Util.padded_video_frame_count(requested_clip_frames),
                    "processed_chunk_input_frames_total": int(
                        sum(chunk.target_input_frame_count for chunk in chunk_plan)
                    ),
                    "audio_present": clip_probe.audio_present,
                    "audio_copied": bool(audio_copy_result.audio_copied) if audio_copy_result else False,
                    "audio_copy_mode": audio_copy_result.copy_mode if audio_copy_result else None,
                    "audio_copy_reason": audio_copy_result.reason if audio_copy_result else "not_attempted",
                    "temporal_chunk_size": temporal_chunk_size,
                    "temporal_chunk_overlap": temporal_chunk_overlap,
                    "temporal_chunk_count": len(chunk_plan),
                    "temporal_chunk_plan": [
                        {
                            "input_start_frame": chunk.input_start_frame,
                            "input_end_frame": chunk.input_end_frame,
                            "input_frame_count": chunk.input_end_frame - chunk.input_start_frame,
                            "target_input_frame_count": chunk.target_input_frame_count,
                            "trim_leading_context_frames": chunk.trim_leading_context_frames,
                            "output_frame_count": chunk.output_frame_count,
                        }
                        for chunk in chunk_plan
                    ],
                    "color_correction_mode": color_correction_mode,
                },
            )
            if validate_health:
                file_health = VideoHealth.validate_file(
                    file_path,
                    expected_width=final_width,
                    expected_height=final_height,
                    expected_frames=requested_clip_frames,
                    expected_fps=clip_probe.fps,
                )
                metadata["video_health"] = {"file": asdict(file_health)}
            if export_json_metadata:
                GeneratedVideo.save_metadata(file_path, metadata)
            return file_path
        except Exception:
            if file_path is not None:
                SeedVR2._cleanup_video_artifacts(file_path)
            raise

    def _restore_video_frames(
        self,
        *,
        seed: int,
        frames: list,
        resolution: int | ScaleFactor,
        softness: float,
        color_correction_mode: str,
        enforce_memory_budget: bool = True,
        noise_latents: mx.array | None = None,
    ) -> tuple[mx.array, int, int, int]:
        processed_video, true_height, true_width = SeedVR2Util.preprocess_video_frames(
            frames=frames,
            resolution=resolution,
            softness=softness,
        )
        if enforce_memory_budget:
            self._assert_video_restore_memory_budget(
                frame_count=len(frames),
                true_height=true_height,
                true_width=true_width,
            )
        processed_video, original_frame_count = SeedVR2Util.pad_video_frames(processed_video)
        padded_frame_count = int(processed_video.shape[2])
        tiling_config = self._effective_tiling_config(
            true_height=true_height,
            true_width=true_width,
            allow_encode_tiling=False,
        )

        config = Config(
            width=true_width,
            height=true_height,
            guidance=1.0,
            num_inference_steps=1,
            scheduler="seedvr2_euler",
            model_config=self.model_config,
            dimension_multiple=2,
        )

        initial_latent = VAEUtil.encode(
            vae=self.vae,
            image=processed_video,
            tiling_config=tiling_config,
            preserve_temporal_axis=True,
        )
        static_condition = SeedVR2LatentCreator.create_condition(encoded_latent=initial_latent)
        mx.eval(static_condition)
        del processed_video
        if noise_latents is None:
            latents = SeedVR2LatentCreator.create_noise_latents(
                seed=seed,
                height=initial_latent.shape[-2],
                width=initial_latent.shape[-1],
                num_frames=initial_latent.shape[2],
                latent_channels=initial_latent.shape[1],
            )
        else:
            if noise_latents.shape != initial_latent.shape:
                raise ValueError(
                    "SeedVR2 streamed video noise slice shape mismatch. "
                    f"expected {tuple(initial_latent.shape)}, got {tuple(noise_latents.shape)}."
                )
            latents = noise_latents
        del initial_latent

        text_embedding = getattr(self, "text_embedding", None)
        txt_pos = (
            SeedVR2TextEmbeddings.prepare_positive(text_embedding)
            if text_embedding is not None
            else SeedVR2TextEmbeddings.load_positive()
        )

        ctx = self.callbacks.start(seed=seed, prompt="", config=config)
        ctx.before_loop(latents)

        for t in config.time_steps:
            model_input = mx.concatenate([latents, static_condition], axis=1)
            noise = self.transformer(
                txt=txt_pos,
                vid=model_input,
                timestep=config.scheduler.timesteps[t],
            )
            latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
            ctx.in_loop(t, latents)
            mx.eval(latents)

        ctx.after_loop(latents)
        del model_input
        del noise

        decoded = VAEUtil.decode(
            vae=self.vae,
            latent=latents,
            tiling_config=tiling_config,
            preserve_temporal_axis=True,
        )
        del latents
        decoded = decoded[:, :, :original_frame_count, :true_height, :true_width]
        if color_correction_mode != "off":
            style_video, _, _ = SeedVR2Util.preprocess_video_frames(
                frames=frames[:original_frame_count],
                resolution=resolution,
                softness=softness,
            )
            style = style_video[:, :, :original_frame_count, :true_height, :true_width]
            decoded = SeedVR2Util.apply_color_correction(
                decoded,
                style,
                mode=color_correction_mode,
            )
            del style
            del style_video
        mx.eval(decoded)
        del static_condition
        mx.clear_cache()
        gc.collect()
        return decoded, true_height, true_width, padded_frame_count

    def _build_streamed_video_noise_latents(
        self,
        *,
        seed: int,
        requested_clip_frames: int,
        source_width: int,
        source_height: int,
        resolution: int | ScaleFactor,
    ) -> mx.array:
        true_height, true_width = self._estimate_output_size(
            source_width=source_width,
            source_height=source_height,
            resolution=resolution,
        )
        latent_height = max(1, true_height // SeedVR2Util.LATENT_SPATIAL_SCALE)
        latent_width = max(1, true_width // SeedVR2Util.LATENT_SPATIAL_SCALE)
        latent_channels = int(getattr(self.vae, "latent_channels", 16))
        return SeedVR2LatentCreator.create_noise_latents(
            seed=seed,
            height=latent_height,
            width=latent_width,
            num_frames=SeedVR2Util.latent_video_frame_count(
                SeedVR2Util.padded_video_frame_count(requested_clip_frames)
            ),
            latent_channels=latent_channels,
        )

    def _streamed_video_noise_slice(
        self,
        *,
        global_noise: mx.array,
        chunk_start_frame: int,
        target_input_frame_count: int,
    ) -> mx.array:
        if chunk_start_frame % 4 != 0:
            raise ValueError(
                "SeedVR2 streamed video chunk start must be aligned to the latent temporal stride. "
                f"Got start_frame={chunk_start_frame}."
            )
        latent_start_frame = chunk_start_frame // 4
        latent_frame_count = SeedVR2Util.latent_video_frame_count(target_input_frame_count)
        latent_end_frame = latent_start_frame + latent_frame_count
        available_end = min(int(global_noise.shape[2]), latent_end_frame)
        noise_slice = global_noise[:, :, latent_start_frame:available_end, :, :]
        if int(noise_slice.shape[2]) == latent_frame_count:
            return noise_slice

        missing_frames = latent_frame_count - int(noise_slice.shape[2])
        if missing_frames <= 0:
            return noise_slice
        if int(noise_slice.shape[2]) == 0:
            raise ValueError("SeedVR2 streamed video noise slice resolved to zero frames.")
        padding = mx.repeat(noise_slice[:, :, -1:, :, :], missing_frames, axis=2)
        return mx.concatenate([noise_slice, padding], axis=2)

    def _seedvr2_metadata(self) -> dict[str, str | None]:
        return {
            "seedvr2_checkpoint_variant": getattr(self, "seedvr2_checkpoint_variant", None),
            "seedvr2_source_layout": getattr(self, "seedvr2_source_layout", None),
        }

    def _effective_tiling_config(
        self,
        *,
        true_height: int,
        true_width: int,
        allow_encode_tiling: bool = True,
    ) -> TilingConfig | None:
        tiling_config = self.tiling_config
        if tiling_config is None:
            return None

        if not allow_encode_tiling and getattr(tiling_config, "vae_encode_tiled", False):
            tiling_config = replace(tiling_config, vae_encode_tiled=False)

        min_pixels = getattr(tiling_config, "vae_decode_auto_tile_min_pixels", None)
        output_pixels = int(true_height) * int(true_width)
        tiles_per_dim = getattr(tiling_config, "vae_decode_tiles_per_dim", None)
        if tiles_per_dim and tiles_per_dim > 1:
            if min_pixels is not None and min_pixels > 0 and output_pixels <= int(min_pixels):
                return replace(tiling_config, vae_decode_tiles_per_dim=0)
            return tiling_config

        if min_pixels is None or min_pixels <= 0:
            return tiling_config

        if output_pixels <= int(min_pixels):
            return tiling_config

        return replace(tiling_config, vae_decode_tiles_per_dim=8)

    def _assert_generate_video_supported(
        self,
        *,
        frame_count: int,
        source_width: int,
        source_height: int,
        resolution: int | ScaleFactor,
    ) -> None:
        true_height, true_width = self._estimate_output_size(
            source_width=source_width,
            source_height=source_height,
            resolution=resolution,
        )
        self._assert_video_restore_memory_budget(
            frame_count=frame_count,
            true_height=true_height,
            true_width=true_width,
        )

    def _assert_video_restore_memory_budget(
        self,
        *,
        frame_count: int,
        true_height: int,
        true_width: int,
    ) -> None:
        overrides = self.model_config.transformer_overrides or {}
        inner_dim = int(overrides.get("vid_dim", 2560))
        text_attention_mode = str(overrides.get("text_attention_mode", "window_pool"))
        resident_weight_bytes = int(getattr(self, "seedvr2_resident_weight_bytes", 0))
        estimated_bytes = SeedVR2Util.estimate_video_restore_working_set_bytes(
            frame_count=SeedVR2Util.padded_video_frame_count(frame_count),
            height=true_height,
            width=true_width,
            inner_dim=inner_dim,
            text_attention_mode=text_attention_mode,
        )
        budget_bytes = SeedVR2Util.host_safe_video_memory_budget_bytes(reserve_bytes=resident_weight_bytes)
        if estimated_bytes > budget_bytes:
            raise ValueError(
                "SeedVR2 video restore chunk exceeds the supported host-safe memory budget. "
                f"estimated={estimated_bytes / (1000**3):.2f} GB, "
                f"budget={budget_bytes / (1000**3):.2f} GB. "
                "Use a smaller chunk, a smaller output size, or an explicit unsafe memory profile."
            )

    def _assert_post_chunk_memory_health(
        self,
        *,
        true_height: int,
        true_width: int,
        frame_count: int,
        enforce_peak_budget: bool = True,
    ) -> None:
        resident_weight_bytes = int(getattr(self, "seedvr2_resident_weight_bytes", 0))
        if enforce_peak_budget:
            budget_bytes = SeedVR2Util.host_safe_video_memory_budget_bytes(reserve_bytes=resident_weight_bytes)
            peak_bytes = SeedVR2Util.mlx_peak_memory_bytes()
            working_peak_bytes = None if peak_bytes is None else max(0, peak_bytes - resident_weight_bytes)
            if working_peak_bytes is not None and working_peak_bytes > budget_bytes:
                raise RuntimeError(
                    "SeedVR2 video restore chunk exceeded the supported host-safe MLX peak memory budget. "
                    f"peak={working_peak_bytes / (1000**3):.2f} GB, budget={budget_bytes / (1000**3):.2f} GB."
                )

        active_bytes = SeedVR2Util.mlx_active_memory_bytes()
        if active_bytes is not None and active_bytes > resident_weight_bytes + SeedVR2Util.VIDEO_RUNTIME_SLACK_BYTES:
            raise RuntimeError(
                "SeedVR2 video restore did not return close to its resident-weight memory baseline after chunk cleanup. "
                f"active={active_bytes / (1000**3):.2f} GB, "
                f"resident={resident_weight_bytes / (1000**3):.2f} GB, "
                f"slack={SeedVR2Util.VIDEO_RUNTIME_SLACK_BYTES / (1000**3):.2f} GB."
            )

        cache_bytes = SeedVR2Util.mlx_cache_memory_bytes()
        if cache_bytes is not None and cache_bytes > SeedVR2Util.VIDEO_RUNTIME_SLACK_BYTES:
            raise RuntimeError(
                "SeedVR2 video restore retained too much MLX cache after chunk cleanup. "
                f"cache={cache_bytes / (1000**3):.2f} GB, "
                f"slack={SeedVR2Util.VIDEO_RUNTIME_SLACK_BYTES / (1000**3):.2f} GB."
            )

    @staticmethod
    def _estimate_output_size(
        *,
        source_width: int,
        source_height: int,
        resolution: int | ScaleFactor,
    ) -> tuple[int, int]:
        min_side = min(source_width, source_height)
        target_res = resolution.get_scaled_value(min_side) if isinstance(resolution, ScaleFactor) else resolution
        scale = float(target_res) / float(min_side)
        width = max(16, (int(source_width * scale) // 2) * 2)
        height = max(16, (int(source_height * scale) // 2) * 2)
        width = max(16, width - (width % 16))
        height = max(16, height - (height % 16))
        return height, width

    @staticmethod
    def _cleanup_video_artifacts(file_path: Path) -> None:
        metadata_path = file_path.with_suffix(".metadata.json")
        if metadata_path.exists():
            metadata_path.unlink()
        if file_path.exists():
            file_path.unlink()

    def save_model(self, path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=path,
            weight_definition=SeedVR2WeightDefinition.for_saving(self.model_config),
        )
