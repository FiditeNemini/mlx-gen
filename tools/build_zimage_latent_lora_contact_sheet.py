from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "docs/assets/validation/zimage-latent-lora-2026-06-24"
SOURCE_IMAGE = ROOT / "docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png"
NO_LORA_IMAGE = ROOT / "validation_outputs/zimage_latent_lora_2026_06_24/zimage_q8_latent_childdraw_no_lora.png"
WITH_LORA_IMAGE = ROOT / "validation_outputs/zimage_latent_lora_2026_06_24/zimage_q8_latent_childdraw_with_lora.png"
WITH_LORA_METADATA = (
    ROOT / "validation_outputs/zimage_latent_lora_2026_06_24/zimage_q8_latent_childdraw_with_lora.metadata.json"
)


@dataclass(frozen=True)
class Panel:
    label: str
    path: Path


class ZImageLatentLoraContactSheet:
    PADDING = 32
    GAP = 24
    LABEL_HEIGHT = 46
    PANEL_BOX = (360, 230)
    BACKGROUND = "#ffffff"
    TEXT = "#111827"
    MUTED = "#4b5563"
    BORDER = "#d6d9de"
    PANEL_FILL = "#f4f6f8"

    @staticmethod
    def main() -> None:
        metadata = json.loads(WITH_LORA_METADATA.read_text())
        prompt = metadata["prompt"]
        output_path = BUNDLE_DIR / "zimage_q8_latent_childdraw_contact_sheet.png"
        panels = (
            Panel("Source image", SOURCE_IMAGE),
            Panel("Latent baseline", NO_LORA_IMAGE),
            Panel("With children's-drawing LoRA", WITH_LORA_IMAGE),
        )

        title_font = ZImageLatentLoraContactSheet._font(42, bold=True)
        body_font = ZImageLatentLoraContactSheet._font(24)
        small_font = ZImageLatentLoraContactSheet._font(21)
        label_font = ZImageLatentLoraContactSheet._font(24, bold=True)

        content_width = (
            len(panels) * ZImageLatentLoraContactSheet.PANEL_BOX[0]
            + (len(panels) - 1) * ZImageLatentLoraContactSheet.GAP
        )
        sheet_width = content_width + (ZImageLatentLoraContactSheet.PADDING * 2)

        dummy = Image.new("RGB", (sheet_width, 200), ZImageLatentLoraContactSheet.BACKGROUND)
        draw = ImageDraw.Draw(dummy)

        header_y = ZImageLatentLoraContactSheet.PADDING
        header_y = ZImageLatentLoraContactSheet._draw_wrapped(
            draw,
            "Z-Image Turbo q8 Latent Children's-Drawing LoRA",
            ZImageLatentLoraContactSheet.PADDING,
            header_y,
            content_width,
            title_font,
            ZImageLatentLoraContactSheet.TEXT,
            spacing=8,
            dry_run=True,
        )
        header_y = ZImageLatentLoraContactSheet._draw_wrapped(
            draw,
            "Model: AbstractFramework/z-image-turbo-8bit   Route: z-image.latent",
            ZImageLatentLoraContactSheet.PADDING,
            header_y + 6,
            content_width,
            body_font,
            ZImageLatentLoraContactSheet.TEXT,
            spacing=6,
            dry_run=True,
        )
        header_y = ZImageLatentLoraContactSheet._draw_wrapped(
            draw,
            "Adapter: ostris/z_image_turbo_childrens_drawings:z_image_turbo_childrens_drawings.safetensors",
            ZImageLatentLoraContactSheet.PADDING,
            header_y + 2,
            content_width,
            small_font,
            ZImageLatentLoraContactSheet.MUTED,
            spacing=6,
            dry_run=True,
        )
        header_y = ZImageLatentLoraContactSheet._draw_wrapped(
            draw,
            "Params: seed=9201   steps=20   image_strength=0.35   size=432x240",
            ZImageLatentLoraContactSheet.PADDING,
            header_y + 2,
            content_width,
            small_font,
            ZImageLatentLoraContactSheet.MUTED,
            spacing=6,
            dry_run=True,
        )
        header_y = ZImageLatentLoraContactSheet._draw_wrapped(
            draw,
            f'Prompt: "{prompt}"',
            ZImageLatentLoraContactSheet.PADDING,
            header_y + 2,
            content_width,
            small_font,
            ZImageLatentLoraContactSheet.MUTED,
            spacing=6,
            dry_run=True,
        )
        header_y = ZImageLatentLoraContactSheet._draw_wrapped(
            draw,
            "Same source, prompt, seed, and latent strength. Only the LoRA changes in the right column.",
            ZImageLatentLoraContactSheet.PADDING,
            header_y + 8,
            content_width,
            small_font,
            ZImageLatentLoraContactSheet.TEXT,
            spacing=6,
            dry_run=True,
        )

        panel_top = header_y + 18
        panel_height = ZImageLatentLoraContactSheet.PANEL_BOX[1] + ZImageLatentLoraContactSheet.LABEL_HEIGHT
        sheet_height = panel_top + panel_height + ZImageLatentLoraContactSheet.PADDING

        sheet = Image.new("RGB", (sheet_width, sheet_height), ZImageLatentLoraContactSheet.BACKGROUND)
        draw = ImageDraw.Draw(sheet)

        header_y = ZImageLatentLoraContactSheet.PADDING
        header_y = ZImageLatentLoraContactSheet._draw_wrapped(
            draw,
            "Z-Image Turbo q8 Latent Children's-Drawing LoRA",
            ZImageLatentLoraContactSheet.PADDING,
            header_y,
            content_width,
            title_font,
            ZImageLatentLoraContactSheet.TEXT,
            spacing=8,
        )
        header_y = ZImageLatentLoraContactSheet._draw_wrapped(
            draw,
            "Model: AbstractFramework/z-image-turbo-8bit   Route: z-image.latent",
            ZImageLatentLoraContactSheet.PADDING,
            header_y + 6,
            content_width,
            body_font,
            ZImageLatentLoraContactSheet.TEXT,
            spacing=6,
        )
        header_y = ZImageLatentLoraContactSheet._draw_wrapped(
            draw,
            "Adapter: ostris/z_image_turbo_childrens_drawings:z_image_turbo_childrens_drawings.safetensors",
            ZImageLatentLoraContactSheet.PADDING,
            header_y + 2,
            content_width,
            small_font,
            ZImageLatentLoraContactSheet.MUTED,
            spacing=6,
        )
        header_y = ZImageLatentLoraContactSheet._draw_wrapped(
            draw,
            "Params: seed=9201   steps=20   image_strength=0.35   size=432x240",
            ZImageLatentLoraContactSheet.PADDING,
            header_y + 2,
            content_width,
            small_font,
            ZImageLatentLoraContactSheet.MUTED,
            spacing=6,
        )
        header_y = ZImageLatentLoraContactSheet._draw_wrapped(
            draw,
            f'Prompt: "{prompt}"',
            ZImageLatentLoraContactSheet.PADDING,
            header_y + 2,
            content_width,
            small_font,
            ZImageLatentLoraContactSheet.MUTED,
            spacing=6,
        )
        ZImageLatentLoraContactSheet._draw_wrapped(
            draw,
            "Same source, prompt, seed, and latent strength. Only the LoRA changes in the right column.",
            ZImageLatentLoraContactSheet.PADDING,
            header_y + 8,
            content_width,
            small_font,
            ZImageLatentLoraContactSheet.TEXT,
            spacing=6,
        )

        x = ZImageLatentLoraContactSheet.PADDING
        for panel in panels:
            ZImageLatentLoraContactSheet._draw_panel(draw, sheet, x, panel_top, panel, label_font)
            x += ZImageLatentLoraContactSheet.PANEL_BOX[0] + ZImageLatentLoraContactSheet.GAP

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path)

    @staticmethod
    def _draw_panel(draw: ImageDraw.ImageDraw, sheet: Image.Image, x: int, y: int, panel: Panel, label_font) -> None:
        label_box = (x, y, x + ZImageLatentLoraContactSheet.PANEL_BOX[0], y + ZImageLatentLoraContactSheet.LABEL_HEIGHT)
        draw.rounded_rectangle(label_box, radius=10, fill=ZImageLatentLoraContactSheet.PANEL_FILL, outline=ZImageLatentLoraContactSheet.BORDER)
        draw.text(
            (x + 16, y + 11),
            panel.label,
            font=label_font,
            fill=ZImageLatentLoraContactSheet.TEXT,
        )

        image_top = y + ZImageLatentLoraContactSheet.LABEL_HEIGHT + 10
        image_box = (
            x,
            image_top,
            x + ZImageLatentLoraContactSheet.PANEL_BOX[0],
            image_top + ZImageLatentLoraContactSheet.PANEL_BOX[1],
        )
        draw.rounded_rectangle(image_box, radius=10, fill=ZImageLatentLoraContactSheet.PANEL_FILL, outline=ZImageLatentLoraContactSheet.BORDER)

        image = Image.open(panel.path).convert("RGB")
        fitted = ImageOps.contain(image, ZImageLatentLoraContactSheet.PANEL_BOX)
        image_x = x + (ZImageLatentLoraContactSheet.PANEL_BOX[0] - fitted.width) // 2
        image_y = image_top + (ZImageLatentLoraContactSheet.PANEL_BOX[1] - fitted.height) // 2
        sheet.paste(fitted, (image_x, image_y))

    @staticmethod
    def _draw_wrapped(
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        width: int,
        font,
        fill: str,
        *,
        spacing: int,
        dry_run: bool = False,
    ) -> int:
        lines = ZImageLatentLoraContactSheet._wrap(text, draw, width, font)
        line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
        if not dry_run:
            draw.multiline_text((x, y), "\n".join(lines), font=font, fill=fill, spacing=spacing)
        return y + len(lines) * line_height + (len(lines) - 1) * spacing

    @staticmethod
    def _wrap(text: str, draw: ImageDraw.ImageDraw, width: int, font) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if draw.textlength(candidate, font=font) <= width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    @staticmethod
    def _font(size: int, *, bold: bool = False):
        candidates = (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Neue.ttc",
        )
        for candidate in candidates:
            path = Path(candidate)
            if path.exists():
                try:
                    return ImageFont.truetype(str(path), size=size)
                except OSError:
                    continue
        return ImageFont.load_default()


if __name__ == "__main__":
    ZImageLatentLoraContactSheet.main()
