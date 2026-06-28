from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "docs/assets/validation/lora-route-expansion-2026-06-22"


@dataclass(frozen=True)
class Panel:
    label: str
    image_path: str


@dataclass(frozen=True)
class SheetConfig:
    output_name: str
    title: str
    model: str
    route: str
    adapter: str
    metadata_path: str
    prompt_path: str
    note: str | None
    panels: tuple[Panel, ...]
    panel_box: tuple[int, int]


class LoraRouteExpansionContactSheets:
    PADDING = 32
    GAP = 24
    HEADER_GAP = 12
    PANEL_LABEL_GAP = 10
    LABEL_HEIGHT = 44
    PANEL_FILL = "#f4f6f8"
    BORDER = "#d6d9de"
    TEXT = "#111827"
    MUTED = "#4b5563"
    ACCENT = "#1f2937"
    BACKGROUND = "#ffffff"

    @staticmethod
    def main() -> None:
        for config in LoraRouteExpansionContactSheets._configs():
            LoraRouteExpansionContactSheets._build_sheet(config)

    @staticmethod
    def _configs() -> tuple[SheetConfig, ...]:
        return (
            SheetConfig(
                output_name="qwen_q8_text_realism_ab_contact_sheet.png",
                title="Qwen Image q8 Text Realism LoRA",
                model="AbstractFramework/qwen-image-8bit",
                route="qwen.text",
                adapter="flymy-ai/qwen-image-realism-lora:flymy_realism.safetensors",
                metadata_path="validation_outputs/lora_route_expansion_2026_06_22/qwen_q8_text_realism_with_lora.metadata.json",
                prompt_path="validation_outputs/lora_route_expansion_2026_06_22/qwen_q8_text_realism_with_lora.metadata.json",
                note="Same seed and prompt. The right column adds only the realism adapter.",
                panels=(
                    Panel("No LoRA baseline", "validation_outputs/lora_route_expansion_2026_06_22/qwen_q8_text_realism_no_lora.png"),
                    Panel("With realism LoRA", "validation_outputs/lora_route_expansion_2026_06_22/qwen_q8_text_realism_with_lora.png"),
                ),
                panel_box=(460, 460),
            ),
            SheetConfig(
                output_name="qwen_q8_latent_studio_cfg_auto_contact_sheet.png",
                title="Qwen Image q8 Latent Studio Realism LoRA",
                model="AbstractFramework/qwen-image-8bit",
                route="qwen.latent",
                adapter="prithivMLmods/Qwen-Image-Studio-Realism:qwen-studio-realism.safetensors",
                metadata_path="validation_outputs/production_qwen_flux2_routes_2026_06_22/qwen_q8_latent_portrait_studio_cfg_auto_with_lora.metadata.json",
                prompt_path="validation_outputs/production_qwen_flux2_routes_2026_06_22/qwen_q8_latent_portrait_studio_cfg_auto_with_lora.metadata.json",
                note="Same source, seed, and prompt. The accepted row keeps the pose and park layout while pushing the illustration toward a photographic portrait.",
                panels=(
                    Panel("Source illustration", "docs/assets/validation/lora-route-expansion-2026-06-22/qwen_q8_latent_source_portrait_illustration.png"),
                    Panel("Latent baseline", "validation_outputs/production_qwen_flux2_routes_2026_06_22/qwen_q8_latent_portrait_studio_cfg_auto_no_lora.png"),
                    Panel("With Studio Realism LoRA", "validation_outputs/production_qwen_flux2_routes_2026_06_22/qwen_q8_latent_portrait_studio_cfg_auto_with_lora.png"),
                ),
                panel_box=(360, 360),
            ),
            SheetConfig(
                output_name="qwen2511_q8_reframe_multi_angle_exact_contact_sheet.png",
                title="Qwen Image Edit 2511 q8 Reframe Multi-Angle LoRA",
                model="AbstractFramework/qwen-image-edit-2511-8bit",
                route="qwen.reframe",
                adapter=(
                    "lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors + "
                    "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:qwen-image-edit-2511-multiple-angles-lora.safetensors"
                ),
                metadata_path="validation_outputs/qwen_flux2_route_completion_2026_06_22/qwen2511_q8_reframe_lightning_plus_multiangle.metadata.json",
                prompt_path="validation_outputs/qwen_flux2_route_completion_2026_06_22/qwen2511_q8_reframe_lightning_plus_multiangle.metadata.json",
                note="The baseline uses the same prompt and the Lightning adapter only. The right column adds the multi-angle LoRA on top of the same exact route.",
                panels=(
                    Panel("Source crop", "docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png"),
                    Panel("Prompt-matched Lightning baseline", "validation_outputs/production_qwen_flux2_routes_2026_06_22/qwen2511_q8_reframe_multiangle_promptmatched_no_lora.png"),
                    Panel("Lightning + multi-angle LoRA", "validation_outputs/qwen_flux2_route_completion_2026_06_22/qwen2511_q8_reframe_lightning_plus_multiangle.png"),
                ),
                panel_box=(360, 250),
            ),
            SheetConfig(
                output_name="qwen2511_q8_outpaint_multiangle_exact_contact_sheet.png",
                title="Qwen Image Edit 2511 q8 Outpaint Multi-Angle LoRA",
                model="AbstractFramework/qwen-image-edit-2511-8bit",
                route="qwen.outpaint",
                adapter=(
                    "lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors + "
                    "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:qwen-image-edit-2511-multiple-angles-lora.safetensors"
                ),
                metadata_path="validation_outputs/qwen_flux2_route_completion_2026_06_22/qwen2511_q8_outpaint_lightning_plus_multiangle.metadata.json",
                prompt_path="validation_outputs/qwen_flux2_route_completion_2026_06_22/qwen2511_q8_outpaint_lightning_plus_multiangle.metadata.json",
                note="The baseline uses the same prompt and the Lightning adapter only. The right column adds the multi-angle LoRA on the same outpaint canvas and seed.",
                panels=(
                    Panel("Source crop", "docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png"),
                    Panel("Prompt-matched Lightning baseline", "validation_outputs/production_qwen_flux2_routes_2026_06_22/qwen2511_q8_outpaint_multiangle_promptmatched_no_lora.png"),
                    Panel("Lightning + multi-angle LoRA", "validation_outputs/qwen_flux2_route_completion_2026_06_22/qwen2511_q8_outpaint_lightning_plus_multiangle.png"),
                ),
                panel_box=(360, 220),
            ),
            SheetConfig(
                output_name="qwen2511_q8_multi_reference_multiangle_exact_contact_sheet.png",
                title="Qwen Image Edit 2511 q8 Multi-Reference Multi-Angle LoRA",
                model="AbstractFramework/qwen-image-edit-2511-8bit",
                route="qwen.multi-reference",
                adapter=(
                    "lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors + "
                    "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:qwen-image-edit-2511-multiple-angles-lora.safetensors"
                ),
                metadata_path="validation_outputs/qwen_flux2_route_completion_2026_06_22/qwen2511_q8_multi_reference_lightning_plus_multiangle.metadata.json",
                prompt_path="validation_outputs/qwen_flux2_route_completion_2026_06_22/qwen2511_q8_multi_reference_lightning_plus_multiangle.metadata.json",
                note="The baseline uses the same prompt and the Lightning adapter only. The right column adds the multi-angle LoRA while keeping the two reference images fixed.",
                panels=(
                    Panel("Reference A: pencil style", "docs/assets/validation/qwen-edit-2511-parity-2026-06-06/qwen2511-source-pencil.png"),
                    Panel("Reference B: crash content", "docs/assets/validation/qwen-edit-2511-parity-2026-06-06/qwen2511-source-crash.png"),
                    Panel("Prompt-matched Lightning baseline", "validation_outputs/production_qwen_flux2_routes_2026_06_22/qwen2511_q8_multi_reference_multiangle_promptmatched_no_lora.png"),
                    Panel("Lightning + multi-angle LoRA", "validation_outputs/qwen_flux2_route_completion_2026_06_22/qwen2511_q8_multi_reference_lightning_plus_multiangle.png"),
                ),
                panel_box=(300, 170),
            ),
            SheetConfig(
                output_name="flux2_klein9b_q8_multiref_exact_contact_sheet.png",
                title="FLUX.2 Klein 9B q8 Multi-Reference Migration LoRA",
                model="AbstractFramework/flux.2-klein-9b-8bit",
                route="flux2.multi-reference",
                adapter="dx8152/Flux2-Klein-9B-Migration:Klein-Migration.safetensors",
                metadata_path="validation_outputs/production_flux2_routes_2026_06_22/flux2_klein9b_q8_multiref_migration.metadata.json",
                prompt_path="validation_outputs/production_flux2_routes_2026_06_22/flux2_klein9b_q8_multiref_migration.metadata.json",
                note="Same references, prompt, and seed. The right column adds only the migration adapter.",
                panels=(
                    Panel("Reference A: pencil crash", "docs/assets/validation/i2i-edit-5x4-2026-06-05/reference-inputs/flux2_klein_9b_8bit_d_pencil_crash.png"),
                    Panel("Reference B: cinematic color", "docs/assets/validation/i2i-edit-5x4-2026-06-05/reference-inputs/flux2_klein_9b_8bit_b_cinematic.png"),
                    Panel("No LoRA multi-reference baseline", "validation_outputs/production_flux2_routes_2026_06_22/flux2_klein9b_q8_multiref_no_lora.png"),
                    Panel("With Migration LoRA", "validation_outputs/production_flux2_routes_2026_06_22/flux2_klein9b_q8_multiref_migration.png"),
                ),
                panel_box=(300, 170),
            ),
            SheetConfig(
                output_name="flux2_klein_base4b_q8_outpaint_route_exact_contact_sheet.png",
                title="FLUX.2 Klein Base 4B q8 Outpaint LoRA",
                model="AbstractFramework/flux.2-klein-base-4b-8bit",
                route="flux2.outpaint",
                adapter="fal/flux-2-klein-4B-outpaint-lora:flux-outpaint-lora.safetensors",
                metadata_path="validation_outputs/production_flux2_routes_2026_06_22/flux2_klein_base4b_q8_outpaint_route_with_lora.metadata.json",
                prompt_path="validation_outputs/production_flux2_routes_2026_06_22/flux2_klein_base4b_q8_outpaint_route_with_lora.metadata.json",
                note="The base route already supports outpaint without a LoRA. This A/B isolates the dedicated outpaint adapter on the same generated green canvas, prompt, and seed.",
                panels=(
                    Panel("Generated outpaint canvas", "validation_outputs/production_flux2_routes_2026_06_22/flux2_klein4b_outpaint_green_canvas.png"),
                    Panel("Base outpaint route", "validation_outputs/production_flux2_routes_2026_06_22/flux2_klein_base4b_q8_outpaint_route_no_lora.png"),
                    Panel("Base route + outpaint LoRA", "validation_outputs/production_flux2_routes_2026_06_22/flux2_klein_base4b_q8_outpaint_route_with_lora.png"),
                ),
                panel_box=(360, 220),
            ),
            SheetConfig(
                output_name="ernie_turbo_q8_latent_anime_style_ab_contact_sheet.png",
                title="ERNIE Image Turbo q8 Latent Anime-Style LoRA",
                model="AbstractFramework/ernie-image-turbo-8bit",
                route="ernie-image.latent",
                adapter="reverentelusarca/ernie-image-elusarca-anime-style-lora:ernie-anime-v1.safetensors",
                metadata_path="validation_outputs/lora_route_expansion_2026_06_22/ernie_turbo_q8_latent_portrait75_with_lora.metadata.json",
                prompt_path="validation_outputs/lora_route_expansion_2026_06_22/ernie_turbo_q8_latent_portrait75_with_lora.metadata.json",
                note="Same source, seed, and prompt. The adapter pushes the portrait cleanly into an anime treatment without changing the setup.",
                panels=(
                    Panel("Source portrait", "docs/assets/validation/lora-route-expansion-2026-06-22/qwen_q8_text_realism_no_lora.png"),
                    Panel("Latent baseline", "validation_outputs/lora_route_expansion_2026_06_22/ernie_turbo_q8_latent_portrait75_no_lora.png"),
                    Panel("With anime-style LoRA", "validation_outputs/lora_route_expansion_2026_06_22/ernie_turbo_q8_latent_portrait75_with_lora.png"),
                ),
                panel_box=(360, 360),
            ),
        )

    @staticmethod
    def _build_sheet(config: SheetConfig) -> None:
        metadata = LoraRouteExpansionContactSheets._load_json(config.metadata_path)
        prompt_metadata = LoraRouteExpansionContactSheets._load_json(config.prompt_path)
        title_font = LoraRouteExpansionContactSheets._font(42, bold=True)
        body_font = LoraRouteExpansionContactSheets._font(24)
        small_font = LoraRouteExpansionContactSheets._font(21)
        label_font = LoraRouteExpansionContactSheets._font(24, bold=True)

        panel_count = len(config.panels)
        content_width = (panel_count * config.panel_box[0]) + ((panel_count - 1) * LoraRouteExpansionContactSheets.GAP)
        sheet_width = content_width + (LoraRouteExpansionContactSheets.PADDING * 2)

        dummy = Image.new("RGB", (sheet_width, 200), LoraRouteExpansionContactSheets.BACKGROUND)
        draw = ImageDraw.Draw(dummy)

        header_height = LoraRouteExpansionContactSheets.PADDING
        header_height = LoraRouteExpansionContactSheets._draw_wrapped_block(
            draw,
            text=config.title,
            x=LoraRouteExpansionContactSheets.PADDING,
            y=header_height,
            width=content_width,
            font=title_font,
            fill=LoraRouteExpansionContactSheets.TEXT,
            spacing=8,
            dry_run=True,
        )
        header_height += 6
        route_line = f"Model: {config.model}   Route: {config.route}"
        header_height = LoraRouteExpansionContactSheets._draw_wrapped_block(
            draw,
            text=route_line,
            x=LoraRouteExpansionContactSheets.PADDING,
            y=header_height,
            width=content_width,
            font=body_font,
            fill=LoraRouteExpansionContactSheets.ACCENT,
            spacing=6,
            dry_run=True,
        )
        adapter_text = f"Adapter: {config.adapter}"
        header_height = LoraRouteExpansionContactSheets._draw_wrapped_block(
            draw,
            text=adapter_text,
            x=LoraRouteExpansionContactSheets.PADDING,
            y=header_height + 2,
            width=content_width,
            font=small_font,
            fill=LoraRouteExpansionContactSheets.MUTED,
            spacing=6,
            dry_run=True,
        )
        params_line = LoraRouteExpansionContactSheets._params_line(metadata)
        header_height = LoraRouteExpansionContactSheets._draw_wrapped_block(
            draw,
            text=f"Params: {params_line}",
            x=LoraRouteExpansionContactSheets.PADDING,
            y=header_height + 2,
            width=content_width,
            font=small_font,
            fill=LoraRouteExpansionContactSheets.MUTED,
            spacing=6,
            dry_run=True,
        )
        prompt_text = f'Prompt: "{prompt_metadata["prompt"]}"'
        header_height = LoraRouteExpansionContactSheets._draw_wrapped_block(
            draw,
            text=prompt_text,
            x=LoraRouteExpansionContactSheets.PADDING,
            y=header_height + 6,
            width=content_width,
            font=body_font,
            fill=LoraRouteExpansionContactSheets.TEXT,
            spacing=8,
            dry_run=True,
        )
        if config.note is not None:
            header_height = LoraRouteExpansionContactSheets._draw_wrapped_block(
                draw,
                text=f"Note: {config.note}",
                x=LoraRouteExpansionContactSheets.PADDING,
                y=header_height + 4,
                width=content_width,
                font=small_font,
                fill=LoraRouteExpansionContactSheets.MUTED,
                spacing=6,
                dry_run=True,
            )

        header_height += LoraRouteExpansionContactSheets.HEADER_GAP
        sheet_height = (
            header_height
            + LoraRouteExpansionContactSheets.LABEL_HEIGHT
            + config.panel_box[1]
            + LoraRouteExpansionContactSheets.PADDING
        )

        sheet = Image.new("RGB", (sheet_width, sheet_height), LoraRouteExpansionContactSheets.BACKGROUND)
        draw = ImageDraw.Draw(sheet)

        y = LoraRouteExpansionContactSheets.PADDING
        y = LoraRouteExpansionContactSheets._draw_wrapped_block(
            draw,
            text=config.title,
            x=LoraRouteExpansionContactSheets.PADDING,
            y=y,
            width=content_width,
            font=title_font,
            fill=LoraRouteExpansionContactSheets.TEXT,
            spacing=8,
        )
        y += 6
        y = LoraRouteExpansionContactSheets._draw_wrapped_block(
            draw,
            text=route_line,
            x=LoraRouteExpansionContactSheets.PADDING,
            y=y,
            width=content_width,
            font=body_font,
            fill=LoraRouteExpansionContactSheets.ACCENT,
            spacing=6,
        )
        y = LoraRouteExpansionContactSheets._draw_wrapped_block(
            draw,
            text=adapter_text,
            x=LoraRouteExpansionContactSheets.PADDING,
            y=y + 2,
            width=content_width,
            font=small_font,
            fill=LoraRouteExpansionContactSheets.MUTED,
            spacing=6,
        )
        y = LoraRouteExpansionContactSheets._draw_wrapped_block(
            draw,
            text=f"Params: {params_line}",
            x=LoraRouteExpansionContactSheets.PADDING,
            y=y + 2,
            width=content_width,
            font=small_font,
            fill=LoraRouteExpansionContactSheets.MUTED,
            spacing=6,
        )
        y = LoraRouteExpansionContactSheets._draw_wrapped_block(
            draw,
            text=prompt_text,
            x=LoraRouteExpansionContactSheets.PADDING,
            y=y + 6,
            width=content_width,
            font=body_font,
            fill=LoraRouteExpansionContactSheets.TEXT,
            spacing=8,
        )
        if config.note is not None:
            y = LoraRouteExpansionContactSheets._draw_wrapped_block(
                draw,
                text=f"Note: {config.note}",
                x=LoraRouteExpansionContactSheets.PADDING,
                y=y + 4,
                width=content_width,
                font=small_font,
                fill=LoraRouteExpansionContactSheets.MUTED,
                spacing=6,
            )

        y += LoraRouteExpansionContactSheets.HEADER_GAP
        x = LoraRouteExpansionContactSheets.PADDING
        for panel in config.panels:
            LoraRouteExpansionContactSheets._draw_panel(
                sheet,
                draw,
                panel=panel,
                box=(x, y, config.panel_box[0], config.panel_box[1]),
                label_font=label_font,
                body_font=small_font,
            )
            x += config.panel_box[0] + LoraRouteExpansionContactSheets.GAP

        output_path = BUNDLE_DIR / config.output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path, quality=95)

    @staticmethod
    def _draw_panel(
        sheet: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        panel: Panel,
        box: tuple[int, int, int, int],
        label_font,
        body_font,
    ) -> None:
        x, y, width, height = box
        label_y = y
        label_lines = LoraRouteExpansionContactSheets._wrap_text(draw, panel.label, label_font, width)
        label_line_height = LoraRouteExpansionContactSheets._line_height(label_font)
        current_y = label_y
        for line in label_lines[:2]:
            draw.text((x, current_y), line, font=label_font, fill=LoraRouteExpansionContactSheets.TEXT)
            current_y += label_line_height + 2

        panel_top = y + LoraRouteExpansionContactSheets.LABEL_HEIGHT
        panel_rect = (x, panel_top, x + width, panel_top + height)
        draw.rounded_rectangle(panel_rect, radius=8, fill=LoraRouteExpansionContactSheets.PANEL_FILL, outline=LoraRouteExpansionContactSheets.BORDER, width=2)

        image = Image.open(ROOT / panel.image_path).convert("RGB")
        image.thumbnail((width - 16, height - 16), Image.Resampling.LANCZOS)
        paste_x = x + ((width - image.width) // 2)
        paste_y = panel_top + ((height - image.height) // 2)
        framed = ImageOps.expand(image, border=2, fill="#ffffff")
        sheet.paste(framed, (paste_x, paste_y))

    @staticmethod
    def _params_line(metadata: dict) -> str:
        parts = [
            f"seed {metadata['seed']}",
            f"steps {metadata['steps']}",
        ]
        guidance = metadata.get("guidance")
        if guidance is not None:
            parts.append(f"guidance {guidance:g}")
        strength = metadata.get("image_strength")
        if strength is not None:
            parts.append(f"strength {strength:g}")
        outpaint_padding = metadata.get("outpaint_padding")
        if outpaint_padding is not None:
            parts.append(f"outpaint {outpaint_padding}")
        reframe_padding = metadata.get("reframe_padding")
        if reframe_padding is not None:
            parts.append(f"reframe {reframe_padding}")
        parts.append(f"{metadata['width']}x{metadata['height']}")
        return " | ".join(parts)

    @staticmethod
    def _draw_wrapped_block(
        draw: ImageDraw.ImageDraw,
        *,
        text: str,
        x: int,
        y: int,
        width: int,
        font,
        fill: str,
        spacing: int,
        dry_run: bool = False,
    ) -> int:
        lines = LoraRouteExpansionContactSheets._wrap_text(draw, text, font, width)
        line_height = LoraRouteExpansionContactSheets._line_height(font)
        current_y = y
        for line in lines:
            if not dry_run:
                draw.text((x, current_y), line, font=font, fill=fill)
            current_y += line_height + spacing
        return current_y

    @staticmethod
    def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
        words = text.split()
        if not words:
            return [""]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
                continue
            lines.append(current)
            current = word
        lines.append(current)
        return lines

    @staticmethod
    def _line_height(font) -> int:
        bbox = font.getbbox("Ag")
        return bbox[3] - bbox[1]

    @staticmethod
    def _load_json(path: str) -> dict:
        return json.loads((ROOT / path).read_text())

    @staticmethod
    def _font(size: int, *, bold: bool = False):
        primary = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        fallback = "Arial Bold.ttf" if bold else "Arial.ttf"
        try:
            return ImageFont.truetype(primary, size=size)
        except OSError:
            try:
                return ImageFont.truetype(fallback, size=size)
            except OSError:
                pass
        return ImageFont.load_default()


if __name__ == "__main__":
    LoraRouteExpansionContactSheets.main()
