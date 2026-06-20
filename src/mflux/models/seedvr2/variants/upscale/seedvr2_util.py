from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.utils.scale_factor import ScaleFactor


class SeedVR2Util:
    @staticmethod
    def preprocess_image(
        image_path: str | Path,
        resolution: int | ScaleFactor,
        softness: float = 0.0,
    ) -> tuple[mx.array, int, int]:
        image = Image.open(image_path).convert("RGB")
        resized, true_h, true_w = SeedVR2Util._resize_and_soften(image=image, resolution=resolution, softness=softness)
        resized = SeedVR2Util._pad_to_multiple(resized, factor=16)
        img_mx = SeedVR2Util._pil_to_mx_image(resized)
        return img_mx, true_h, true_w

    @staticmethod
    def preprocess_video_frames(
        frames: list[Image.Image],
        resolution: int | ScaleFactor,
        softness: float = 0.0,
    ) -> tuple[mx.array, int, int]:
        if not frames:
            raise ValueError("preprocess_video_frames requires at least one frame.")

        first = frames[0].convert("RGB")
        resized_first, _, _ = SeedVR2Util._resize_and_soften(
            image=first,
            resolution=resolution,
            softness=softness,
        )
        cropped_first = SeedVR2Util._center_crop_to_multiple(resized_first, factor=16)
        true_w, true_h = cropped_first.size

        processed_frames = [SeedVR2Util._pil_to_numpy_video_frame(cropped_first)]
        for frame in frames[1:]:
            rgb_frame = frame.convert("RGB")
            resized, _, _ = SeedVR2Util._resize_and_soften(
                image=rgb_frame,
                resolution=resolution,
                softness=softness,
            )
            cropped = SeedVR2Util._center_crop_to_multiple(resized, factor=16)
            if cropped.size != (true_w, true_h):
                cropped = cropped.resize((true_w, true_h), Image.Resampling.BICUBIC)
            processed_frames.append(SeedVR2Util._pil_to_numpy_video_frame(cropped))

        video_np = np.stack(processed_frames, axis=0)
        video_mx = mx.array(video_np, dtype=mx.float32)
        video_mx = mx.transpose(video_mx, (3, 0, 1, 2))
        video_mx = video_mx[None, ...]
        return video_mx, true_h, true_w

    @staticmethod
    def apply_color_correction(
        content: mx.array,
        style: mx.array,
        mode: str = "lab",
        luminance_weight: float = 0.8,
    ) -> mx.array:
        if mode == "off":
            return content
        if mode == "wavelet":
            return SeedVR2Util._apply_wavelet_color_reconstruction(content=content, style=style)
        if mode != "lab":
            raise ValueError(f"Unsupported SeedVR2 color correction mode: {mode}")
        if content.ndim == 5 and style.ndim == 5:
            return SeedVR2Util._apply_video_color_correction(
                content=content,
                style=style,
                luminance_weight=luminance_weight,
            )
        return SeedVR2Util._lab_color_transfer_exact(content, style, luminance_weight=luminance_weight)

    @staticmethod
    def _apply_video_color_correction(
        content: mx.array,
        style: mx.array,
        luminance_weight: float = 0.8,
    ) -> mx.array:
        if content.shape != style.shape:
            raise ValueError(f"Video color correction requires same shapes, got {content.shape} vs {style.shape}")

        batch, channels, frames, height, width = content.shape
        content_4d = mx.transpose(content, (0, 2, 1, 3, 4)).reshape(batch * frames, channels, height, width)
        style_4d = mx.transpose(style, (0, 2, 1, 3, 4)).reshape(batch * frames, channels, height, width)
        corrected = SeedVR2Util._lab_color_transfer_exact(
            content_4d,
            style_4d,
            luminance_weight=luminance_weight,
        )
        corrected = corrected.reshape(batch, frames, channels, height, width)
        return mx.transpose(corrected, (0, 2, 1, 3, 4))

    @staticmethod
    def _apply_wavelet_color_reconstruction(content: mx.array, style: mx.array) -> mx.array:
        if content.shape != style.shape:
            raise ValueError(f"Wavelet reconstruction requires same shapes, got {content.shape} vs {style.shape}")

        if content.ndim == 5:
            batch, channels, frames, height, width = content.shape
            content_4d = mx.transpose(content, (0, 2, 1, 3, 4)).reshape(batch * frames, channels, height, width)
            style_4d = mx.transpose(style, (0, 2, 1, 3, 4)).reshape(batch * frames, channels, height, width)
            reconstructed = SeedVR2Util._apply_wavelet_color_reconstruction(content_4d, style_4d)
            reconstructed = reconstructed.reshape(batch, frames, channels, height, width)
            return mx.transpose(reconstructed, (0, 2, 1, 3, 4))

        content_np = np.array(content.astype(mx.float32), dtype=np.float32)
        style_np = np.array(style.astype(mx.float32), dtype=np.float32)
        reconstructed = SeedVR2Util._wavelet_reconstruction(content_np, style_np)
        return mx.array(reconstructed, dtype=content.dtype)

    @staticmethod
    def pad_video_frames(video: mx.array) -> tuple[mx.array, int]:
        if video.ndim != 5:
            raise ValueError(f"Expected video tensor [B, C, T, H, W], got {video.shape}")

        frame_count = int(video.shape[2])
        if frame_count == 1:
            return video, frame_count
        if (frame_count - 1) % 4 == 0:
            return video, frame_count

        pad_frames = 4 - ((frame_count - 1) % 4)
        last_frame = video[:, :, -1:, :, :]
        padding = mx.repeat(last_frame, pad_frames, axis=2)
        return mx.concatenate([video, padding], axis=2), frame_count

    @staticmethod
    def padded_video_frame_count(frame_count: int) -> int:
        if frame_count <= 0:
            raise ValueError("frame_count must be greater than zero.")
        if frame_count == 1 or (frame_count - 1) % 4 == 0:
            return frame_count
        return frame_count + (4 - ((frame_count - 1) % 4))

    @staticmethod
    def plan_video_chunks(frame_count: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
        if frame_count <= 0:
            raise ValueError("frame_count must be greater than zero.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if overlap < 0:
            raise ValueError("overlap must be greater than or equal to zero.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        chunks: list[tuple[int, int]] = []
        start = 0
        step = chunk_size - overlap
        while start < frame_count:
            end = min(start + chunk_size, frame_count)
            chunks.append((start, end))
            if end >= frame_count:
                break
            start += step
        return chunks

    @staticmethod
    def blend_overlapping_frames(
        existing_tail: list[Image.Image],
        incoming_head: list[Image.Image],
    ) -> list[Image.Image]:
        if len(existing_tail) != len(incoming_head):
            raise ValueError("SeedVR2 overlap blending requires equal-length frame lists.")
        if not existing_tail:
            return []

        blended: list[Image.Image] = []
        total = len(existing_tail)
        for index, (left, right) in enumerate(zip(existing_tail, incoming_head)):
            alpha = float(index + 1) / float(total + 1)
            left_np = np.array(left.convert("RGB"), dtype=np.float32)
            right_np = np.array(right.convert("RGB"), dtype=np.float32)
            merged = ((1.0 - alpha) * left_np) + (alpha * right_np)
            blended.append(Image.fromarray(np.clip(merged, 0.0, 255.0).round().astype(np.uint8), mode="RGB"))
        return blended

    @staticmethod
    def _lab_color_transfer_exact(content: mx.array, style: mx.array, luminance_weight: float = 0.8) -> mx.array:
        content_f = content.astype(mx.float32)
        style_f = style.astype(mx.float32)

        content_np = np.array(content_f, dtype=np.float32)
        style_np = np.array(style_f, dtype=np.float32)

        content_np = SeedVR2Util._wavelet_reconstruction(content_np, style_np)

        c = np.transpose(content_np, (0, 2, 3, 1))
        s = np.transpose(style_np, (0, 2, 3, 1))
        c = np.clip((c + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
        s = np.clip((s + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)

        c_lab = SeedVR2Util._rgb_to_lab(c)
        s_lab = SeedVR2Util._rgb_to_lab(s)

        matched_a = SeedVR2Util._hist_match(c_lab[..., 1], s_lab[..., 1])
        matched_b = SeedVR2Util._hist_match(c_lab[..., 2], s_lab[..., 2])

        if luminance_weight < 1.0:
            matched_L = SeedVR2Util._hist_match(c_lab[..., 0], s_lab[..., 0])
            L = luminance_weight * c_lab[..., 0] + (1.0 - luminance_weight) * matched_L
        else:
            L = c_lab[..., 0]

        out_lab = np.stack([L, matched_a, matched_b], axis=-1)
        out_rgb = SeedVR2Util._lab_to_rgb(out_lab)
        out_rgb = np.clip(out_rgb, 0.0, 1.0)

        out = out_rgb * 2.0 - 1.0
        out = mx.array(out, dtype=mx.float32)
        out = mx.transpose(out, (0, 3, 1, 2))
        return out.astype(content.dtype)

    @staticmethod
    def _wavelet_blur(image: np.ndarray, radius: int) -> np.ndarray:
        if radius < 1:
            radius = 1

        h, w = int(image.shape[-2]), int(image.shape[-1])
        max_safe_radius = max(1, min(h, w) // 8)
        if radius > max_safe_radius:
            radius = max_safe_radius

        kernel = np.array(
            [
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625],
            ],
            dtype=np.float32,
        )

        p = radius
        padded = np.pad(image, ((0, 0), (0, 0), (p, p), (p, p)), mode="edge")

        out = np.zeros_like(image, dtype=np.float32)
        H, W = image.shape[-2], image.shape[-1]

        for ky, dy in enumerate((-1, 0, 1)):
            ys = p + dy * radius
            ye = ys + H
            for kx, dx in enumerate((-1, 0, 1)):
                xs = p + dx * radius
                xe = xs + W
                out += kernel[ky, kx] * padded[:, :, ys:ye, xs:xe]

        return out

    @staticmethod
    def _wavelet_decomposition(image: np.ndarray, levels: int = 5) -> tuple[np.ndarray, np.ndarray]:
        high_freq = np.zeros_like(image, dtype=np.float32)
        cur = image.astype(np.float32)

        for i in range(levels):
            radius = 2**i
            low_freq = SeedVR2Util._wavelet_blur(cur, radius)
            high_freq += cur - low_freq
            cur = low_freq

        return high_freq, cur

    @staticmethod
    def _wavelet_reconstruction(content: np.ndarray, style: np.ndarray) -> np.ndarray:
        if content.shape != style.shape:
            raise ValueError(f"Wavelet reconstruction requires same shapes, got {content.shape} vs {style.shape}")

        content_high, _ = SeedVR2Util._wavelet_decomposition(content, levels=5)
        _, style_low = SeedVR2Util._wavelet_decomposition(style, levels=5)
        return np.clip(content_high + style_low, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)

    @staticmethod
    def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.0031308, 1.055 * np.maximum(x, 0.0) ** (1.0 / 2.4) - 0.055, 12.92 * x)

    @staticmethod
    def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
        rgb_lin = SeedVR2Util._srgb_to_linear(rgb.astype(np.float32))
        M = np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ],
            dtype=np.float32,
        )
        xyz = np.tensordot(rgb_lin, M.T, axes=([3], [0]))

        xyz[..., 0] /= 0.95047
        xyz[..., 2] /= 1.08883

        eps = 6.0 / 29.0
        eps3 = eps**3
        kappa = (29.0 / 3.0) ** 3

        f = np.where(xyz > eps3, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
        fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]

        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        return np.stack([L, a, b], axis=-1).astype(np.float32)

    @staticmethod
    def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0

        eps = 6.0 / 29.0
        kappa = (29.0 / 3.0) ** 3

        x = np.where(fx > eps, fx**3, (116.0 * fx - 16.0) / kappa)
        y = np.where(fy > eps, fy**3, (116.0 * fy - 16.0) / kappa)
        z = np.where(fz > eps, fz**3, (116.0 * fz - 16.0) / kappa)

        x *= 0.95047
        z *= 1.08883

        xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
        M_inv = np.array(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ],
            dtype=np.float32,
        )
        rgb_lin = np.tensordot(xyz, M_inv.T, axes=([3], [0]))
        rgb = SeedVR2Util._linear_to_srgb(rgb_lin)
        return rgb.astype(np.float32)

    @staticmethod
    def _hist_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        out = np.empty_like(source, dtype=np.float32)
        B = source.shape[0]
        for i in range(B):
            src = source[i].reshape(-1).astype(np.float32)
            ref = reference[i].reshape(-1).astype(np.float32)
            src_idx = np.argsort(src, kind="stable")
            ref_sorted = np.sort(ref, kind="stable")
            inv = np.argsort(src_idx, kind="stable")
            out[i] = ref_sorted[inv].reshape(source.shape[1:]).astype(np.float32)
        return out

    @staticmethod
    def _resize_and_soften(
        *,
        image: Image.Image,
        resolution: int | ScaleFactor,
        softness: float,
    ) -> tuple[Image.Image, int, int]:
        w, h = image.size
        if isinstance(resolution, ScaleFactor):
            target_res = resolution.get_scaled_value(min(w, h))
        else:
            target_res = resolution

        scale = target_res / min(w, h)
        true_w = max(2, (int(w * scale) // 2) * 2)
        true_h = max(2, (int(h * scale) // 2) * 2)
        factor = 1.0 + (max(0.0, min(1.0, softness)) * 7.0)

        if factor <= 1.0 and true_w == w and true_h == h:
            return image.copy(), true_h, true_w

        if factor > 1.0:
            down_w = max(2, int(true_w / factor))
            down_h = max(2, int(true_h / factor))
            down = image.resize((down_w, down_h), Image.Resampling.BICUBIC)
            resized = down.resize((true_w, true_h), Image.Resampling.BICUBIC)
        else:
            resized = image.resize((true_w, true_h), Image.Resampling.BICUBIC)

        return resized, true_h, true_w

    @staticmethod
    def _pad_to_multiple(image: Image.Image, *, factor: int) -> Image.Image:
        width, height = image.size
        pad_w = (factor - (width % factor)) % factor
        pad_h = (factor - (height % factor)) % factor
        if pad_w == 0 and pad_h == 0:
            return image

        padded = Image.new("RGB", (width + pad_w, height + pad_h), (0, 0, 0))
        padded.paste(image, (0, 0))
        return padded

    @staticmethod
    def _center_crop_to_multiple(image: Image.Image, *, factor: int) -> Image.Image:
        width, height = image.size
        cropped_width = width - (width % factor)
        cropped_height = height - (height % factor)
        left = max((width - cropped_width) // 2, 0)
        top = max((height - cropped_height) // 2, 0)
        return image.crop((left, top, left + cropped_width, top + cropped_height))

    @staticmethod
    def _pil_to_mx_image(image: Image.Image) -> mx.array:
        img_mx = mx.array(np.array(image)).astype(mx.float32) / 255.0
        img_mx = mx.clip(img_mx, 0.0, 1.0)
        img_mx = img_mx * 2.0 - 1.0
        img_mx = mx.transpose(img_mx, (2, 0, 1))
        return img_mx[None, ...]

    @staticmethod
    def _pil_to_numpy_video_frame(image: Image.Image) -> np.ndarray:
        frame_np = np.asarray(image, dtype=np.float32) / 255.0
        frame_np = np.clip(frame_np, 0.0, 1.0)
        return frame_np * 2.0 - 1.0
