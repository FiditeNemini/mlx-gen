from dataclasses import dataclass
from pathlib import Path

import PIL.Image
import PIL.ImageChops
import PIL.ImageFilter
import PIL.ImageOps
import PIL.ImageStat

from mflux.utils.box_values import AbsoluteBoxValues, BoxValues
from mflux.utils.image_util import ImageUtil


@dataclass(frozen=True)
class OutpaintCanvas:
    canvas_path: Path
    source_path: Path
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    paste_left: int
    paste_top: int
    padding: AbsoluteBoxValues


class OutpaintUtil:
    @staticmethod
    def create_expanded_canvas(
        *,
        source_path: str | Path,
        padding_value: str,
        output_path: str | Path,
        dimension_multiple: int = 16,
        fill_color: tuple[int, int, int] = (255, 255, 255),
        fill_mode: str = "edge",
        option_name: str = "--outpaint-padding",
    ) -> OutpaintCanvas:
        if dimension_multiple <= 0:
            raise ValueError("dimension_multiple must be greater than zero.")

        source = ImageUtil.load_image(source_path)
        padding = BoxValues.parse(padding_value).normalize_to_dimensions(width=source.width, height=source.height)
        OutpaintUtil._validate_padding(padding, option_name=option_name)

        target_width = OutpaintUtil._round_up(source.width + padding.left + padding.right, dimension_multiple)
        target_height = OutpaintUtil._round_up(source.height + padding.top + padding.bottom, dimension_multiple)
        canvas = OutpaintUtil._create_background(
            source=source,
            target_width=target_width,
            target_height=target_height,
            paste_left=padding.left,
            paste_top=padding.top,
            fill_color=fill_color,
            fill_mode=fill_mode,
        )
        canvas.paste(source, (padding.left, padding.top))

        canvas_path = Path(output_path)
        canvas_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(canvas_path)
        return OutpaintCanvas(
            canvas_path=canvas_path,
            source_path=Path(source_path),
            source_width=source.width,
            source_height=source.height,
            target_width=target_width,
            target_height=target_height,
            paste_left=padding.left,
            paste_top=padding.top,
            padding=padding,
        )

    @staticmethod
    def composite_source_region(
        *,
        generated_image: PIL.Image.Image,
        canvas: OutpaintCanvas,
        feather_px: int | None = None,
        restore_threshold: float = 12.0,
    ) -> PIL.Image.Image:
        if generated_image.size != (canvas.target_width, canvas.target_height):
            raise ValueError(
                "Outpaint output size changed unexpectedly: "
                f"expected {canvas.target_width}x{canvas.target_height}, got "
                f"{generated_image.width}x{generated_image.height}."
            )

        source = ImageUtil.load_image(canvas.source_path)
        if source.size != (canvas.source_width, canvas.source_height):
            raise ValueError("Source image size changed during outpaint generation.")

        composited = generated_image.convert("RGB").copy()
        restore_difference = OutpaintUtil._source_region_difference(
            generated_image=composited,
            source=source,
            paste_left=canvas.paste_left,
            paste_top=canvas.paste_top,
        )
        if feather_px is None and restore_difference > restore_threshold:
            composited.outpaint_preservation_applied = False
            composited.outpaint_source_restore_difference = restore_difference
            return composited

        composited.paste(
            source,
            (canvas.paste_left, canvas.paste_top),
            OutpaintUtil._source_mask(source=source, feather_px=feather_px),
        )
        composited.outpaint_preservation_applied = True
        composited.outpaint_source_restore_difference = restore_difference
        return composited

    @staticmethod
    def attach_metadata(
        *,
        generated_image,
        canvas: OutpaintCanvas,
        padding_value: str,
        preservation: str = "adaptive-content-aware-source-blend",
    ) -> None:
        generated_image.source_image_width = canvas.source_width
        generated_image.source_image_height = canvas.source_height
        extra_metadata = dict(getattr(generated_image, "extra_metadata", None) or {})
        output_image = getattr(generated_image, "image", None)
        extra_metadata.update(
            {
                "outpaint_padding": padding_value,
                "outpaint_source_path": str(canvas.source_path),
                "outpaint_target_width": canvas.target_width,
                "outpaint_target_height": canvas.target_height,
                "outpaint_source_paste_left": canvas.paste_left,
                "outpaint_source_paste_top": canvas.paste_top,
                "outpaint_preservation": preservation,
                "outpaint_source_restore_applied": getattr(
                    output_image,
                    "outpaint_preservation_applied",
                    None,
                ),
                "outpaint_source_restore_difference": getattr(
                    output_image,
                    "outpaint_source_restore_difference",
                    None,
                ),
            }
        )
        generated_image.extra_metadata = extra_metadata

    @staticmethod
    def attach_reframe_metadata(*, generated_image, canvas: OutpaintCanvas, padding_value: str) -> None:
        generated_image.source_image_width = canvas.source_width
        generated_image.source_image_height = canvas.source_height
        extra_metadata = dict(getattr(generated_image, "extra_metadata", None) or {})
        extra_metadata.update(
            {
                "reframe_padding": padding_value,
                "reframe_source_path": str(canvas.source_path),
                "reframe_target_width": canvas.target_width,
                "reframe_target_height": canvas.target_height,
                "reframe_source_paste_left": canvas.paste_left,
                "reframe_source_paste_top": canvas.paste_top,
                "reframe_mode": "expanded-conditioning-canvas",
            }
        )
        generated_image.extra_metadata = extra_metadata

    @staticmethod
    def _validate_padding(padding: AbsoluteBoxValues, *, option_name: str) -> None:
        parts = (padding.top, padding.right, padding.bottom, padding.left)
        if any(part < 0 for part in parts):
            raise ValueError(f"{option_name} values must be zero or positive.")
        if not any(part > 0 for part in parts):
            raise ValueError(f"{option_name} must add pixels on at least one side.")

    @staticmethod
    def _round_up(value: int, multiple: int) -> int:
        remainder = value % multiple
        if remainder == 0:
            return value
        return value + multiple - remainder

    @staticmethod
    def _create_background(
        *,
        source: PIL.Image.Image,
        target_width: int,
        target_height: int,
        paste_left: int,
        paste_top: int,
        fill_color: tuple[int, int, int],
        fill_mode: str,
    ) -> PIL.Image.Image:
        if fill_mode == "solid":
            return PIL.Image.new("RGB", (target_width, target_height), fill_color)
        if fill_mode == "edge":
            return OutpaintUtil._create_edge_extended_background(
                source=source,
                target_width=target_width,
                target_height=target_height,
                paste_left=paste_left,
                paste_top=paste_top,
                fill_color=fill_color,
            )
        if fill_mode != "blur":
            raise ValueError("fill_mode must be 'edge', 'blur', or 'solid'.")

        background = PIL.ImageOps.fit(
            source,
            (target_width, target_height),
            method=PIL.Image.Resampling.BICUBIC,
            centering=(0.5, 0.5),
        )
        blur_radius = max(target_width, target_height) / 24
        return background.filter(PIL.ImageFilter.GaussianBlur(radius=blur_radius)).convert("RGB")

    @staticmethod
    def _create_edge_extended_background(
        *,
        source: PIL.Image.Image,
        target_width: int,
        target_height: int,
        paste_left: int,
        paste_top: int,
        fill_color: tuple[int, int, int],
    ) -> PIL.Image.Image:
        source = source.convert("RGB")
        canvas = PIL.Image.new("RGB", (target_width, target_height), fill_color)
        left = paste_left
        top = paste_top
        right = max(0, target_width - paste_left - source.width)
        bottom = max(0, target_height - paste_top - source.height)
        strip_x = max(1, min(32, source.width // 8))
        strip_y = max(1, min(32, source.height // 8))

        if left > 0:
            canvas.paste(
                OutpaintUtil._resized_patch(
                    source.crop((0, 0, strip_x, source.height)),
                    (left, source.height),
                ),
                (0, top),
            )
        if right > 0:
            canvas.paste(
                OutpaintUtil._resized_patch(
                    source.crop((source.width - strip_x, 0, source.width, source.height)),
                    (right, source.height),
                ),
                (paste_left + source.width, top),
            )
        if top > 0:
            canvas.paste(
                OutpaintUtil._resized_patch(
                    source.crop((0, 0, source.width, strip_y)),
                    (source.width, top),
                ),
                (left, 0),
            )
        if bottom > 0:
            canvas.paste(
                OutpaintUtil._resized_patch(
                    source.crop((0, source.height - strip_y, source.width, source.height)),
                    (source.width, bottom),
                ),
                (left, paste_top + source.height),
            )

        OutpaintUtil._paste_corner_extensions(
            canvas=canvas,
            source=source,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            strip_x=strip_x,
            strip_y=strip_y,
        )
        border = max(left, top, right, bottom)
        if border <= 0:
            return canvas
        blur_radius = min(8, max(2, border // 24))
        return canvas.filter(PIL.ImageFilter.GaussianBlur(radius=blur_radius))

    @staticmethod
    def _paste_corner_extensions(
        *,
        canvas: PIL.Image.Image,
        source: PIL.Image.Image,
        left: int,
        top: int,
        right: int,
        bottom: int,
        strip_x: int,
        strip_y: int,
    ) -> None:
        if left > 0 and top > 0:
            canvas.paste(
                OutpaintUtil._resized_patch(source.crop((0, 0, strip_x, strip_y)), (left, top)),
                (0, 0),
            )
        if right > 0 and top > 0:
            canvas.paste(
                OutpaintUtil._resized_patch(
                    source.crop((source.width - strip_x, 0, source.width, strip_y)),
                    (right, top),
                ),
                (left + source.width, 0),
            )
        if left > 0 and bottom > 0:
            canvas.paste(
                OutpaintUtil._resized_patch(
                    source.crop((0, source.height - strip_y, strip_x, source.height)),
                    (left, bottom),
                ),
                (0, top + source.height),
            )
        if right > 0 and bottom > 0:
            canvas.paste(
                OutpaintUtil._resized_patch(
                    source.crop((source.width - strip_x, source.height - strip_y, source.width, source.height)),
                    (right, bottom),
                ),
                (left + source.width, top + source.height),
            )

    @staticmethod
    def _resized_patch(patch: PIL.Image.Image, size: tuple[int, int]) -> PIL.Image.Image:
        return patch.resize(size, resample=PIL.Image.Resampling.BICUBIC)

    @staticmethod
    def _source_mask(source: PIL.Image.Image, feather_px: int | None) -> PIL.Image.Image:
        if feather_px is None:
            if min(source.width, source.height) < 64:
                return PIL.Image.new("L", source.size, 255)
            return OutpaintUtil._content_aware_source_mask(source=source)
        if feather_px <= 0:
            return PIL.Image.new("L", source.size, 255)

        inset_x = min(feather_px, max(0, source.width // 2 - 1))
        inset_y = min(feather_px, max(0, source.height // 2 - 1))
        mask = PIL.Image.new("L", source.size, 0)
        if inset_x == 0 or inset_y == 0:
            return PIL.Image.new("L", source.size, 255)
        mask.paste(255, (inset_x, inset_y, source.width - inset_x, source.height - inset_y))
        mask = mask.filter(PIL.ImageFilter.GaussianBlur(radius=feather_px / 2))
        exact_inset_x = min(feather_px * 2, max(0, source.width // 2 - 1))
        exact_inset_y = min(feather_px * 2, max(0, source.height // 2 - 1))
        if exact_inset_x > 0 and exact_inset_y > 0:
            mask.paste(
                255,
                (
                    exact_inset_x,
                    exact_inset_y,
                    source.width - exact_inset_x,
                    source.height - exact_inset_y,
                ),
            )
        return PIL.ImageChops.lighter(mask, OutpaintUtil._detail_preservation_mask(source=source))

    @staticmethod
    def _content_aware_source_mask(source: PIL.Image.Image) -> PIL.Image.Image:
        feather_px = min(96, max(32, min(source.width, source.height) // 3))
        inset_x = min(feather_px, max(0, source.width // 2 - 1))
        inset_y = min(feather_px, max(0, source.height // 2 - 1))
        mask = PIL.Image.new("L", source.size, 0)
        if inset_x > 0 and inset_y > 0:
            mask.paste(220, (inset_x, inset_y, source.width - inset_x, source.height - inset_y))
            mask = mask.filter(PIL.ImageFilter.GaussianBlur(radius=feather_px / 2))
        detail_mask = OutpaintUtil._detail_preservation_mask(
            source=source,
            border_fade_px=max(8, feather_px // 3),
        )
        return PIL.ImageChops.lighter(mask, detail_mask)

    @staticmethod
    def _source_region_difference(
        *,
        generated_image: PIL.Image.Image,
        source: PIL.Image.Image,
        paste_left: int,
        paste_top: int,
    ) -> float:
        generated_region = generated_image.crop(
            (
                paste_left,
                paste_top,
                paste_left + source.width,
                paste_top + source.height,
            )
        )
        sample_width = min(96, source.width)
        sample_height = max(1, round(source.height * sample_width / source.width))
        source_sample = source.resize((sample_width, sample_height), resample=PIL.Image.Resampling.BICUBIC)
        generated_sample = generated_region.resize(
            (sample_width, sample_height),
            resample=PIL.Image.Resampling.BICUBIC,
        )
        stat = PIL.ImageStat.Stat(PIL.ImageChops.difference(source_sample, generated_sample))
        return float(sum(stat.mean) / len(stat.mean))

    @staticmethod
    def _detail_preservation_mask(source: PIL.Image.Image, border_fade_px: int = 0) -> PIL.Image.Image:
        edges = PIL.ImageOps.grayscale(source).filter(PIL.ImageFilter.FIND_EDGES)
        edges.paste(0, (0, 0, edges.width, 1))
        edges.paste(0, (0, edges.height - 1, edges.width, edges.height))
        edges.paste(0, (0, 0, 1, edges.height))
        edges.paste(0, (edges.width - 1, 0, edges.width, edges.height))
        edges = edges.filter(PIL.ImageFilter.MaxFilter(size=3)).filter(PIL.ImageFilter.GaussianBlur(radius=1))
        if border_fade_px > 0:
            fade = PIL.Image.new("L", source.size, 0)
            inset_x = min(border_fade_px, max(0, source.width // 2 - 1))
            inset_y = min(border_fade_px, max(0, source.height // 2 - 1))
            if inset_x > 0 and inset_y > 0:
                fade.paste(255, (inset_x, inset_y, source.width - inset_x, source.height - inset_y))
                fade = fade.filter(PIL.ImageFilter.GaussianBlur(radius=border_fade_px / 2))
                edges = PIL.ImageChops.multiply(edges, fade)
        return edges.point(lambda value: 255 if value > 18 else 0)
