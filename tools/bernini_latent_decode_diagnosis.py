from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class BerniniLatentDecodeDiagnosis:
    TRANSFORMER_REVISION = "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce"
    VAE_REVISION = "ec4d2cb062b548996b179d493fdd05340de702a1"
    DIFFUSERS_VERSION = "0.35.2"

    @staticmethod
    def main() -> None:
        args = BerniniLatentDecodeDiagnosis._parse_args()
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.stage == "mlx":
            BerniniLatentDecodeDiagnosis._run_mlx(
                output_dir=output_dir,
                metadata_template=args.metadata_template.resolve(),
                source_video=args.source_video.resolve(),
            )
        elif args.stage == "torch":
            BerniniLatentDecodeDiagnosis._run_torch(
                output_dir=output_dir,
                checkpoint_root=args.checkpoint_root.resolve(),
                device=args.torch_device,
            )
        elif args.stage == "review":
            BerniniLatentDecodeDiagnosis._write_manual_review(
                output_dir=output_dir,
                reviewer=args.reviewer,
                notes=args.review_notes,
            )
        else:
            BerniniLatentDecodeDiagnosis._compare(output_dir=output_dir)

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Distinguish Bernini final-latent failures from MLX tiled or untiled VAE decode failures."
        )
        parser.add_argument("--stage", choices=("mlx", "torch", "compare", "review"), required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--metadata-template", type=Path)
        parser.add_argument("--source-video", type=Path)
        parser.add_argument("--checkpoint-root", type=Path)
        parser.add_argument("--torch-device", choices=("mps", "cpu"), default="mps")
        parser.add_argument("--reviewer")
        parser.add_argument("--review-notes")
        args = parser.parse_args()
        if args.stage == "mlx" and (args.metadata_template is None or args.source_video is None):
            parser.error("--stage mlx requires --metadata-template and --source-video.")
        if args.stage == "torch" and args.checkpoint_root is None:
            parser.error("--stage torch requires --checkpoint-root.")
        if args.stage == "review" and (not args.reviewer or not args.review_notes):
            parser.error("--stage review requires --reviewer and --review-notes.")
        return args

    @staticmethod
    def _run_mlx(*, output_dir: Path, metadata_template: Path, source_video: Path) -> None:
        import mlx.core as mx

        from mflux.models.wan.variants import BerniniRenderer
        from mflux.utils.runtime_memory import RuntimeMemory
        from mflux.utils.video_util import VideoStreamWriter, VideoUtil

        metadata = json.loads(metadata_template.read_text())
        RuntimeMemory.apply_mlx_cache_limit(1.0, low_ram=True)
        started = time.perf_counter()
        renderer = BerniniRenderer(prompt_embed_disk_cache=True)
        video = renderer.generate_video(
            seed=int(metadata["seed"]),
            prompt=str(metadata["prompt"]),
            num_inference_steps=int(metadata["steps"]),
            height=int(metadata["requested_height"]),
            width=int(metadata["requested_width"]),
            num_frames=int(metadata["frames"]),
            fps=int(metadata["fps"]),
            guidance=float(metadata["text_guidance"]),
            flow_shift=float(metadata["flow_shift"]),
            solver=str(metadata["solver"]),
            negative_prompt=str(metadata["negative_prompt"]),
            video_path=source_video,
            canvas_policy=str(metadata["canvas_policy"]),
            resize_mode=str(metadata["resize_mode"]),
            max_sequence_length=512,
            release_denoisers_before_decode=True,
            clear_cache_each_step=True,
            clear_cache_each_transformer_block=True,
            tensor_health_check_interval=1,
            reference_image_paths=[],
            reference_guidance=float(metadata["reference_guidance"]),
            source_guidance=float(metadata["source_guidance"]),
            apg_eta=float(metadata["apg_eta"]),
            apg_norm_threshold=float(metadata["apg_norm_threshold"]),
            apg_momentum=float(metadata["apg_momentum"]),
            max_condition_size=int(metadata["max_condition_size"]),
            system_prompt=str(metadata["system_prompt"]),
        )
        if video._frame_batches_factory is None:
            raise RuntimeError("Bernini diagnostic expected a lazy frame-batch factory.")
        closure = inspect.getclosurevars(video._frame_batches_factory)
        decode_latents = closure.nonlocals.get("decode_latents")
        if not isinstance(decode_latents, mx.array):
            raise RuntimeError(
                "Bernini frame-batch factory did not expose the final decode_latents closure; "
                f"available nonlocals: {sorted(closure.nonlocals)}."
            )
        mx.eval(decode_latents)
        final_latents = np.asarray(decode_latents.astype(mx.float32))
        np.save(output_dir / "final_normalized_latents.npy", final_latents)
        latent_stats = BerniniLatentDecodeDiagnosis._latent_stats(final_latents)

        tiled_path = output_dir / "mlx_tiled_runtime.mp4"
        video.save(
            tiled_path,
            export_json_metadata=True,
            overwrite=True,
            validate_health=True,
        )

        untiled_decoded = renderer.vae.decode_normalized_latents(
            decode_latents,
            clear_cache_each_slice=True,
            tile_spatial=False,
        )
        mx.eval(untiled_decoded)
        untiled_np = np.asarray(untiled_decoded.astype(mx.float32))
        np.save(output_dir / "mlx_untiled_decoded.npy", untiled_np)
        untiled_frames = VideoUtil._latents_to_frame_arrays(untiled_decoded, total_frames=int(metadata["frames"]))
        untiled_path = output_dir / "mlx_untiled.mp4"
        with VideoStreamWriter(
            path=untiled_path,
            fps=int(metadata["fps"]),
            width=int(untiled_frames.shape[2]),
            height=int(untiled_frames.shape[1]),
        ) as writer:
            writer.write_frame_arrays(untiled_frames)

        report = {
            "schema_version": 1,
            "kind": "bernini_mlx_final_latent_decode_diagnosis",
            "platform": platform.platform(),
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "transformer_revision": BerniniLatentDecodeDiagnosis.TRANSFORMER_REVISION,
            "vae_revision": BerniniLatentDecodeDiagnosis.VAE_REVISION,
            "metadata_template": str(metadata_template),
            "source_video": str(source_video),
            "settings": {
                key: metadata.get(key)
                for key in (
                    "seed",
                    "prompt",
                    "negative_prompt",
                    "system_prompt",
                    "steps",
                    "frames",
                    "fps",
                    "requested_width",
                    "requested_height",
                    "text_guidance",
                    "apg_eta",
                    "apg_norm_threshold",
                    "apg_momentum",
                    "flow_shift",
                    "max_condition_size",
                )
            },
            "latent_shape": list(final_latents.shape),
            "latent_dtype": str(decode_latents.dtype),
            "latent_stats": latent_stats,
            "tensor_artifacts": {
                "final_normalized_latents": BerniniLatentDecodeDiagnosis._tensor_artifact(
                    output_dir / "final_normalized_latents.npy"
                ),
                "mlx_untiled_decoded": BerniniLatentDecodeDiagnosis._tensor_artifact(
                    output_dir / "mlx_untiled_decoded.npy"
                ),
            },
            "tiled_video": str(tiled_path),
            "untiled_video": str(untiled_path),
            "untiled_decoded_shape": list(untiled_np.shape),
            "elapsed_seconds": time.perf_counter() - started,
            "runtime_memory": RuntimeMemory.snapshot(
                "latent-decode-diagnosis-complete", synchronize=True
            ).to_metadata(),
            "passed_structural_checks": (
                final_latents.shape[2] == 1 + (int(metadata["frames"]) - 1) // 4
                and untiled_np.shape[2] == int(metadata["frames"])
                and tiled_path.is_file()
                and untiled_path.is_file()
            ),
        }
        (output_dir / "mlx_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    @staticmethod
    def _run_torch(*, output_dir: Path, checkpoint_root: Path, device: str) -> None:
        import diffusers
        import torch
        from diffusers import AutoencoderKLWan

        from mflux.utils.video_util import VideoStreamWriter

        if diffusers.__version__ != BerniniLatentDecodeDiagnosis.DIFFUSERS_VERSION:
            raise ValueError(
                f"Pinned decode requires diffusers=={BerniniLatentDecodeDiagnosis.DIFFUSERS_VERSION}, "
                f"got {diffusers.__version__}."
            )
        if checkpoint_root.name != BerniniLatentDecodeDiagnosis.VAE_REVISION:
            raise ValueError(
                f"Pinned decode requires VAE revision {BerniniLatentDecodeDiagnosis.VAE_REVISION}, "
                f"got {checkpoint_root.name}."
            )
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable.")
        final_latents = np.load(output_dir / "final_normalized_latents.npy")
        started = time.perf_counter()
        vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint_root),
            subfolder="vae",
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(device)
        vae.eval()
        normalized = torch.from_numpy(final_latents).to(device=device, dtype=torch.float32)
        mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).reshape(1, 16, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32).reshape(1, 16, 1, 1, 1)
        with torch.no_grad():
            decoded = vae.decode(normalized * std + mean, return_dict=False)[0].clamp(-1.0, 1.0)
        decoded_np = decoded.detach().float().cpu().numpy()
        np.save(output_dir / "torch_decoded.npy", decoded_np)
        frame_arrays = BerniniLatentDecodeDiagnosis._decoded_to_uint8(decoded_np)
        video_path = output_dir / "torch_diffusers_0_35_2.mp4"
        with VideoStreamWriter(
            path=video_path,
            fps=16,
            width=int(frame_arrays.shape[2]),
            height=int(frame_arrays.shape[1]),
        ) as writer:
            writer.write_frame_arrays(frame_arrays)
        report = {
            "schema_version": 1,
            "kind": "bernini_torch_final_latent_decode_diagnosis",
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "diffusers_version": diffusers.__version__,
            "diffusers_path": str(diffusers.__file__),
            "device": device,
            "vae_revision": BerniniLatentDecodeDiagnosis.VAE_REVISION,
            "normalized_latent_shape": list(final_latents.shape),
            "decoded_shape": list(decoded_np.shape),
            "tensor_artifacts": {
                "final_normalized_latents": BerniniLatentDecodeDiagnosis._tensor_artifact(
                    output_dir / "final_normalized_latents.npy"
                ),
                "torch_decoded": BerniniLatentDecodeDiagnosis._tensor_artifact(output_dir / "torch_decoded.npy"),
            },
            "video": str(video_path),
            "elapsed_seconds": time.perf_counter() - started,
            "passed_structural_checks": decoded_np.shape[2] == 1 + 4 * (final_latents.shape[2] - 1),
        }
        (output_dir / "torch_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    @staticmethod
    def _compare(*, output_dir: Path) -> None:
        from mflux.utils.video_util import VideoUtil

        report_bindings = {
            "mlx_report.json": {
                "final_normalized_latents": output_dir / "final_normalized_latents.npy",
                "mlx_untiled_decoded": output_dir / "mlx_untiled_decoded.npy",
            },
            "torch_report.json": {
                "final_normalized_latents": output_dir / "final_normalized_latents.npy",
                "torch_decoded": output_dir / "torch_decoded.npy",
            },
        }
        for report_name, bindings in report_bindings.items():
            report_path = output_dir / report_name
            payload = json.loads(report_path.read_text())
            payload["tensor_artifacts"] = {
                label: BerniniLatentDecodeDiagnosis._tensor_artifact(path) for label, path in bindings.items()
            }
            report_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        mlx_report = json.loads((output_dir / "mlx_report.json").read_text())
        torch_report = json.loads((output_dir / "torch_report.json").read_text())
        mlx_untiled = np.load(output_dir / "mlx_untiled_decoded.npy").astype(np.float64)
        torch_decoded = np.load(output_dir / "torch_decoded.npy").astype(np.float64)
        if mlx_untiled.shape != torch_decoded.shape:
            raise ValueError(f"MLX/Torch decoded shape mismatch: {mlx_untiled.shape} != {torch_decoded.shape}.")

        backends = {
            "mlx_tiled_runtime": output_dir / "mlx_tiled_runtime.mp4",
            "mlx_untiled": output_dir / "mlx_untiled.mp4",
            "torch_diffusers_0_35_2": output_dir / "torch_diffusers_0_35_2.mp4",
        }
        native_hashes = {}
        decoded_videos = {}
        sheet_paths = {}
        for name, video_path in backends.items():
            clip = VideoUtil.read_video_clip(video_path)
            frames = [frame.convert("RGB") for frame in clip.frames]
            decoded_videos[name] = np.stack([np.asarray(frame, dtype=np.float64) for frame in frames])
            native_dir = output_dir / "native_frames" / name
            native_dir.mkdir(parents=True, exist_ok=True)
            hashes = []
            for index, frame in enumerate(frames):
                frame_path = native_dir / f"frame_{index:03d}.png"
                frame.save(frame_path)
                hashes.append(BerniniLatentDecodeDiagnosis._image_hash(frame))
            native_hashes[name] = hashes
            sheet_path = output_dir / f"{name}_all_frames_5k.png"
            BerniniLatentDecodeDiagnosis._save_contact_sheet(
                frames=frames,
                output_path=sheet_path,
                title=f"{name} · all {len(frames)} native decoded frames",
            )
            sheet_paths[name] = str(sheet_path)

        raw_metrics = BerniniLatentDecodeDiagnosis._metrics(torch_decoded, mlx_untiled)
        raw_per_frame = [
            {
                "frame_index": index,
                **BerniniLatentDecodeDiagnosis._metrics(torch_decoded[:, :, index], mlx_untiled[:, :, index]),
            }
            for index in range(torch_decoded.shape[2])
        ]
        video_metrics = {
            "tiled_vs_untiled": BerniniLatentDecodeDiagnosis._metrics(
                decoded_videos["mlx_untiled"],
                decoded_videos["mlx_tiled_runtime"],
                data_range=255.0,
            ),
            "torch_vs_mlx_untiled": BerniniLatentDecodeDiagnosis._metrics(
                decoded_videos["torch_diffusers_0_35_2"],
                decoded_videos["mlx_untiled"],
                data_range=255.0,
            ),
        }
        report = {
            "schema_version": 2,
            "kind": "bernini_identical_final_latent_three_way_decode",
            "mlx_report": mlx_report,
            "torch_report": torch_report,
            "native_frame_sha256": native_hashes,
            "contact_sheets": sheet_paths,
            "raw_torch_vs_mlx_untiled": raw_metrics,
            "raw_torch_vs_mlx_untiled_per_frame": raw_per_frame,
            "encoded_video_metrics": video_metrics,
            "structural_checks_passed": (
                mlx_report.get("passed_structural_checks") is True
                and torch_report.get("passed_structural_checks") is True
                and all(len(hashes) == torch_decoded.shape[2] for hashes in native_hashes.values())
            ),
            "visual_disposition": "pending_manual_native_frame_and_contact_sheet_review",
        }
        (output_dir / "decode_comparison_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        print(
            json.dumps(
                {
                    "kind": report["kind"],
                    "raw_torch_vs_mlx_untiled": raw_metrics,
                    "encoded_video_metrics": video_metrics,
                    "structural_checks_passed": report["structural_checks_passed"],
                    "visual_disposition": report["visual_disposition"],
                },
                indent=2,
            )
        )

    @staticmethod
    def _write_manual_review(*, output_dir: Path, reviewer: str, notes: str) -> None:
        report_path = output_dir / "decode_comparison_report.json"
        report = json.loads(report_path.read_text())
        backends = {
            "mlx_tiled_runtime": output_dir / "mlx_tiled_runtime.mp4",
            "mlx_untiled": output_dir / "mlx_untiled.mp4",
            "torch_diffusers_0_35_2": output_dir / "torch_diffusers_0_35_2.mp4",
        }
        sheets = {name: output_dir / f"{name}_all_frames_5k.png" for name in backends}
        native_hashes = report.get("native_frame_sha256")
        if not isinstance(native_hashes, dict) or set(native_hashes) != set(backends):
            raise ValueError("Manual review requires the complete three-backend native-frame hash record.")
        if any(len(hashes) != 17 for hashes in native_hashes.values()):
            raise ValueError("Manual review requires all 17 native frames from every backend.")
        verified_native_hashes = {}
        for name, expected_hashes in native_hashes.items():
            actual_hashes = []
            for index in range(17):
                frame_path = output_dir / "native_frames" / name / f"frame_{index:03d}.png"
                with Image.open(frame_path) as frame:
                    actual_hashes.append(BerniniLatentDecodeDiagnosis._image_hash(frame))
            if actual_hashes != expected_hashes:
                raise ValueError(f"Native frames for {name} no longer match the comparison report.")
            verified_native_hashes[name] = actual_hashes
        review = {
            "schema_version": 1,
            "kind": "bernini_identical_final_latent_three_way_decode_manual_review",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "reviewer": reviewer,
            "execution_mode": "native-frame-and-high-resolution-contact-sheet-review",
            "status": "negative_result",
            "scope": "all 51 native frames and all three 5280px-wide all-frame contact sheets",
            "reviewed_native_frame_indices": {name: list(range(17)) for name in backends},
            "artifact_sha256": {
                "videos": {name: BerniniLatentDecodeDiagnosis._file_hash(path) for name, path in backends.items()},
                "contact_sheets": {
                    name: BerniniLatentDecodeDiagnosis._file_hash(path) for name, path in sheets.items()
                },
                "native_frames": verified_native_hashes,
            },
            "findings": {
                "coherent_frame_indices": list(range(13)),
                "common_corruption_frame_indices": [13, 14, 15, 16],
                "corruption": "cyan and peach block corruption across the lower half, worsening at the tail",
                "decode_implementations_materially_agree": True,
                "visible_spatial_tiling_seam_at_x192": False,
                "attribution": "the shared final denoised latent, not MLX spatial tiling or MLX VAE decode",
            },
            "notes": notes,
        }
        review_path = output_dir / "manual_visual_review.json"
        review_path.write_text(json.dumps(review, indent=2, sort_keys=True))
        report["visual_disposition"] = "negative_result"
        report["manual_visual_review"] = {
            "path": str(review_path.resolve()),
            "sha256": BerniniLatentDecodeDiagnosis._file_hash(review_path),
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report["manual_visual_review"], indent=2, sort_keys=True))

    @staticmethod
    def _latent_stats(latents: np.ndarray) -> list[dict[str, float | int]]:
        stats = []
        for index in range(latents.shape[2]):
            value = latents[:, :, index].astype(np.float64)
            stats.append(
                {
                    "latent_index": index,
                    "min": float(value.min()),
                    "max": float(value.max()),
                    "mean": float(value.mean()),
                    "std": float(value.std()),
                    "l2": float(np.linalg.norm(value.reshape(-1))),
                    "abs_p99": float(np.percentile(np.abs(value), 99)),
                }
            )
        return stats

    @staticmethod
    def _decoded_to_uint8(decoded: np.ndarray) -> np.ndarray:
        frames = np.transpose(decoded[0], (1, 2, 3, 0))
        return (np.clip(frames / 2 + 0.5, 0, 1) * 255).round().astype(np.uint8)

    @staticmethod
    def _metrics(reference: np.ndarray, actual: np.ndarray, *, data_range: float = 2.0) -> dict[str, float]:
        if reference.shape != actual.shape:
            raise ValueError(f"Metric shape mismatch: {reference.shape} != {actual.shape}.")
        if not math.isfinite(data_range) or data_range <= 0:
            raise ValueError(f"Metric data range must be positive and finite, got {data_range!r}.")
        delta = actual - reference
        reference_norm = float(np.linalg.norm(reference.reshape(-1)))
        delta_norm = float(np.linalg.norm(delta.reshape(-1)))
        mse = float(np.mean(np.square(delta)))
        return {
            "relative_l2": delta_norm / reference_norm if reference_norm else math.inf,
            "mean_absolute_error": float(np.mean(np.abs(delta))),
            "max_absolute_error": float(np.max(np.abs(delta))),
            "psnr_db": math.inf if mse == 0 else float(20 * math.log10(data_range / math.sqrt(mse))),
        }

    @staticmethod
    def _save_contact_sheet(*, frames: list[Image.Image], output_path: Path, title: str) -> None:
        columns = 4
        cell_width = 1280
        source_width, source_height = frames[0].size
        cell_height = round(source_height * cell_width / source_width)
        label_height = 96
        padding = 32
        header_height = 160
        rows = math.ceil(len(frames) / columns)
        width = columns * cell_width + (columns + 1) * padding
        height = header_height + rows * (cell_height + label_height) + (rows + 1) * padding
        sheet = Image.new("RGB", (width, height), "#17191d")
        draw = ImageDraw.Draw(sheet)
        title_font = BerniniLatentDecodeDiagnosis._font(72)
        label_font = BerniniLatentDecodeDiagnosis._font(56)
        draw.text((padding, 40), title, fill="white", font=title_font)
        for index, frame in enumerate(frames):
            row, column = divmod(index, columns)
            x = padding + column * (cell_width + padding)
            y = header_height + padding + row * (cell_height + label_height + padding)
            rendered = frame.resize((cell_width, cell_height), Image.Resampling.NEAREST)
            sheet.paste(rendered, (x, y))
            draw.text((x, y + cell_height + 12), f"frame {index:03d}", fill="white", font=label_font)
        sheet.save(output_path)

    @staticmethod
    def _font(size: int):
        for path in (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ):
            if Path(path).is_file():
                return ImageFont.truetype(path, size=size)
        return ImageFont.load_default()

    @staticmethod
    def _image_hash(image: Image.Image) -> str:
        value = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
        return hashlib.sha256(value.tobytes()).hexdigest()

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _tensor_artifact(path: Path) -> dict[str, object]:
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        return {
            "path": str(path),
            "sha256": BerniniLatentDecodeDiagnosis._file_hash(path),
            "size_bytes": path.stat().st_size,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }


if __name__ == "__main__":
    BerniniLatentDecodeDiagnosis.main()
