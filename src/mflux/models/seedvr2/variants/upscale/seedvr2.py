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
from mflux.utils.video_util import VideoStreamWriter, VideoUtil


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

        return ImageUtil.to_image(
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
        video_clip = VideoUtil.read_video_clip(
            path=video_path,
            start_seconds=start_seconds,
            max_frames=max_frames,
        )
        decoded, true_height, true_width, padded_input_frames = self._restore_video_frames(
            seed=seed,
            frames=video_clip.frames,
            resolution=resolution,
            softness=softness,
            color_correction_mode=color_correction_mode,
        )
        generation_time = time.perf_counter() - start_time

        return VideoUtil.to_video(
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
                "source_clip_frames": video_clip.clip_frame_count,
                "padded_input_frames": padded_input_frames,
                "audio_present": video_clip.audio_present,
                "audio_copied": False,
                "temporal_chunk_size": None,
                "temporal_chunk_overlap": None,
                "color_correction_mode": color_correction_mode,
            },
        )

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
        restore_metadata: dict | None = None,
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
        chunk_plan = SeedVR2Util.plan_video_chunks(
            frame_count=requested_clip_frames,
            chunk_size=temporal_chunk_size,
            overlap=temporal_chunk_overlap,
        )

        pending_overlap_frames: list | None = None
        final_width: int | None = None
        final_height: int | None = None
        writer = None
        try:
            chunk_clips = VideoUtil.iter_video_frame_windows(
                video_path,
                start_frame=clip_probe.clip_start_frame,
                windows=chunk_plan,
            )
            for chunk_index, chunk_clip in enumerate(chunk_clips):
                chunk_frames = chunk_clip.frames
                start_frame, end_frame = chunk_plan[chunk_index]
                decoded, true_height, true_width, _ = self._restore_video_frames(
                    seed=seed,
                    frames=chunk_frames,
                    resolution=resolution,
                    softness=softness,
                    color_correction_mode=color_correction_mode,
                )
                restored_frames = VideoUtil._latents_to_frames(decoded)
                final_width = true_width
                final_height = true_height

                next_overlap = 0
                if chunk_index + 1 < len(chunk_plan):
                    next_start, _ = chunk_plan[chunk_index + 1]
                    next_overlap = max(0, end_frame - next_start)

                if pending_overlap_frames is not None:
                    current_overlap = len(pending_overlap_frames)
                    blended_frames = SeedVR2Util.blend_overlapping_frames(
                        existing_tail=pending_overlap_frames,
                        incoming_head=restored_frames[:current_overlap],
                    )
                    body_start = current_overlap
                else:
                    blended_frames = []
                    body_start = 0

                body_end = len(restored_frames) - next_overlap if next_overlap > 0 else len(restored_frames)
                batch_to_write = blended_frames + restored_frames[body_start:body_end]
                if batch_to_write:
                    if writer is None:
                        writer = VideoStreamWriter(
                            path=output_path,
                            fps=clip_probe.fps,
                            width=batch_to_write[0].width,
                            height=batch_to_write[0].height,
                            overwrite=overwrite,
                        )
                    writer.write_frames(batch_to_write)

                pending_overlap_frames = restored_frames[-next_overlap:] if next_overlap > 0 else None

                del chunk_frames
                del batch_to_write
                del blended_frames
                del decoded
                del restored_frames
                mx.clear_cache()
                gc.collect()

            if pending_overlap_frames:
                if writer is None:
                    writer = VideoStreamWriter(
                        path=output_path,
                        fps=clip_probe.fps,
                        width=pending_overlap_frames[0].width,
                        height=pending_overlap_frames[0].height,
                        overwrite=overwrite,
                    )
                writer.write_frames(pending_overlap_frames)

            if writer is None:
                raise ValueError("SeedVR2 video restore did not produce any frames.")
            file_path = writer.close()
        except Exception:
            if writer is not None:
                writer.abort()
            raise

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
                "source_clip_frames": requested_clip_frames,
                "padded_input_frames": SeedVR2Util.padded_video_frame_count(requested_clip_frames),
                "processed_chunk_input_frames_total": int(sum(end - start for start, end in chunk_plan)),
                "audio_present": clip_probe.audio_present,
                "audio_copied": False,
                "temporal_chunk_size": temporal_chunk_size,
                "temporal_chunk_overlap": temporal_chunk_overlap,
                "temporal_chunk_count": len(chunk_plan),
                "temporal_chunk_plan": [
                    {"start_frame": start, "end_frame": end, "frame_count": end - start}
                    for start, end in chunk_plan
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

    def _restore_video_frames(
        self,
        *,
        seed: int,
        frames: list,
        resolution: int | ScaleFactor,
        softness: float,
        color_correction_mode: str,
    ) -> tuple[mx.array, int, int, int]:
        processed_video, true_height, true_width = SeedVR2Util.preprocess_video_frames(
            frames=frames,
            resolution=resolution,
            softness=softness,
        )
        processed_video, original_frame_count = SeedVR2Util.pad_video_frames(processed_video)
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
        latents = SeedVR2LatentCreator.create_noise_latents(
            seed=seed,
            height=initial_latent.shape[-2],
            width=initial_latent.shape[-1],
            num_frames=initial_latent.shape[2],
            latent_channels=initial_latent.shape[1],
        )

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

        decoded = VAEUtil.decode(
            vae=self.vae,
            latent=latents,
            tiling_config=tiling_config,
            preserve_temporal_axis=True,
        )
        decoded = decoded[:, :, :original_frame_count, :true_height, :true_width]
        style = processed_video[:, :, :original_frame_count, :true_height, :true_width]
        decoded = SeedVR2Util.apply_color_correction(
            decoded,
            style,
            mode=color_correction_mode,
        )
        return decoded, true_height, true_width, int(processed_video.shape[2])

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

        tiles_per_dim = getattr(tiling_config, "vae_decode_tiles_per_dim", None)
        if tiles_per_dim and tiles_per_dim > 1:
            return tiling_config

        min_pixels = getattr(tiling_config, "vae_decode_auto_tile_min_pixels", None)
        if min_pixels is None or min_pixels <= 0:
            return tiling_config

        output_pixels = int(true_height) * int(true_width)
        if output_pixels <= int(min_pixels):
            return tiling_config

        return replace(tiling_config, vae_decode_tiles_per_dim=8)

    def save_model(self, path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=path,
            weight_definition=SeedVR2WeightDefinition.for_saving(self.model_config),
        )
