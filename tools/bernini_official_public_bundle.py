from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont


class BerniniOfficialPublicBundle:
    PAD = 28
    GAP = 20
    LABEL_H = 32
    TITLE_H = 44
    BG = (250, 250, 250)
    FG = (20, 20, 20)
    INPUT_COLUMNS = 4
    OUTPUT_COLUMNS = 3
    MAX_CELL_HEIGHT = 1600

    @classmethod
    def main(cls) -> None:
        args = cls._parse_args()
        case_dirs = [Path(case_dir) for case_dir in args.case_dir]
        out_root = Path(args.out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        for case_dir in case_dirs:
            proof = json.loads((case_dir / "proof.json").read_text())
            case_out_dir = out_root / str(proof["id"])
            case_out_dir.mkdir(parents=True, exist_ok=True)
            prompt = cls._prompt_from_case_json(Path(str(proof["case_json"])))
            input_sheet = case_out_dir / "input_sheet.png"
            official_sheet = case_out_dir / "official_sheet.png"
            mlx_sheet = case_out_dir / "mlx_sheet.png"
            cls._ensure_sheet(
                source=case_dir / "input_sheet.png",
                destination=input_sheet,
                rebuild=lambda: cls._build_sheet(
                    items=cls._input_items(
                        image_path=cls._optional_path(proof.get("image_path")),
                        video_path=cls._optional_path(proof.get("video_path")),
                        reference_video_paths=[Path(path) for path in proof.get("reference_video_paths", [])],
                        reference_image_paths=[Path(path) for path in proof.get("reference_image_paths", [])],
                    ),
                    title="Input",
                    columns=cls.INPUT_COLUMNS,
                ),
            )
            official_media = Path(str(proof["official_output"]))
            output_media = Path(str(proof["output"]))
            official_spec = cls._inspect_media(official_media)
            output_spec = cls._inspect_media(output_media)
            cls._ensure_sheet(
                source=case_dir / "official_sheet.png",
                destination=official_sheet,
                rebuild=lambda: cls._build_sheet(
                    items=cls._output_items(media_path=official_media, media_spec=official_spec),
                    title="Official reference output",
                    columns=cls.OUTPUT_COLUMNS,
                ),
            )
            cls._ensure_sheet(
                source=case_dir / "mlx_sheet.png",
                destination=mlx_sheet,
                rebuild=lambda: cls._build_sheet(
                    items=cls._output_items(media_path=output_media, media_spec=output_spec),
                    title="mlx-gen output",
                    columns=cls.OUTPUT_COLUMNS,
                ),
            )
            cls._copy_if_exists(case_dir / "proof.json", case_out_dir / "proof.json")
            cls._copy_if_exists(output_media, case_out_dir / output_media.name)
            cls._copy_if_exists(output_media.with_suffix(".metadata.json"), case_out_dir / output_media.with_suffix(".metadata.json").name)
            cls._write_case_readme(
                case_out_dir=case_out_dir,
                proof=proof,
                prompt=prompt,
                input_sheet=input_sheet.name,
                official_sheet=official_sheet.name,
                mlx_sheet=mlx_sheet.name,
            )
            summary_rows.append(
                {
                    "id": str(proof["id"]),
                    "title": str(proof["title"]),
                    "expected": list(proof.get("expected", [])),
                    "case_dir": case_out_dir.name,
                    "artifact": (case_out_dir / output_media.name).name,
                }
            )
        cls._write_summary_readme(out_root=out_root, summary_rows=summary_rows)

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--out-dir", required=True)
        parser.add_argument("--case-dir", action="append", required=True)
        return parser.parse_args()

    @staticmethod
    def _optional_path(value: str | None) -> Path | None:
        if value is None:
            return None
        return Path(str(value))

    @staticmethod
    def _display_path(value: str | Path | None) -> str:
        if value is None:
            return ""
        path = Path(str(value))
        text = str(path)
        testcase_marker = f"{os.sep}assets{os.sep}testcases{os.sep}"
        if testcase_marker in text:
            return "assets/testcases/" + text.split(testcase_marker, 1)[1].replace(os.sep, "/")
        try:
            relative = path.relative_to(Path.cwd())
        except ValueError:
            return text
        return str(relative).replace(os.sep, "/")

    @classmethod
    def _display_reproduce_command(cls, command: str) -> str:
        official_root = "/private/tmp/bernini_official_20260810"
        return command.replace(official_root, "/path/to/bernini-official")

    @staticmethod
    def _prompt_from_case_json(case_json_path: Path) -> str:
        return str(json.loads(case_json_path.read_text())["prompt"]).strip()

    @staticmethod
    def _copy_if_exists(source: Path, destination: Path) -> None:
        if source.exists():
            shutil.copy2(source, destination)

    @staticmethod
    def _ensure_sheet(*, source: Path, destination: Path, rebuild) -> None:
        if source.exists():
            shutil.copy2(source, destination)
            return
        rebuild().save(destination)

    @classmethod
    def _write_summary_readme(cls, *, out_root: Path, summary_rows: list[dict]) -> None:
        lines = [
            "# Bernini-R 1.3B official public example bundle",
            "",
            "This bundle contains high-resolution per-case proof pages for the accepted official public examples.",
            "",
            "## Cases",
            "",
        ]
        for row in summary_rows:
            lines.extend(
                [
                    f"- [{row['id']}](./{row['case_dir']}/README.md) — {row['title']}",
                    f"  - output: [`{row['artifact']}`](./{row['case_dir']}/{row['artifact']})",
                ]
            )
        lines.append("")
        (out_root / "README.md").write_text("\n".join(lines))

    @classmethod
    def _write_case_readme(
        cls,
        *,
        case_out_dir: Path,
        proof: dict,
        prompt: str,
        input_sheet: str,
        official_sheet: str,
        mlx_sheet: str,
    ) -> None:
        inputs = []
        if proof.get("image_path") is not None:
            inputs.append(f"- source image: `{cls._display_path(proof['image_path'])}`")
        if proof.get("video_path") is not None:
            inputs.append(f"- source video: `{cls._display_path(proof['video_path'])}`")
        for index, path in enumerate(proof.get("reference_video_paths", [])):
            inputs.append(f"- reference video {index + 1}: `{cls._display_path(path)}`")
        for index, path in enumerate(proof.get("reference_image_paths", [])):
            inputs.append(f"- reference image {index + 1}: `{cls._display_path(path)}`")
        if not inputs:
            inputs.append("- no input media")
        expected_lines = "\n".join(f"- {line}" for line in proof.get("expected", []))
        observed = proof.get("observed_result")
        if isinstance(observed, list) and observed:
            observed_lines = "\n".join(f"- {line}" for line in observed)
        elif isinstance(observed, str) and observed.strip():
            observed_lines = observed.strip()
        else:
            observed_lines = "- not yet manually reviewed"
        output_name = Path(str(proof["output"])).name
        metadata_name = Path(str(proof["metadata"])).name
        reproduce_command = str(proof.get("reproduce_command") or "").strip()
        reproduce_section = ""
        if reproduce_command:
            displayed_command = cls._display_reproduce_command(reproduce_command)
            reproduce_section = (
                "## Reproduce\n\n"
                "```bash\n"
                f"{displayed_command}\n"
                "```\n\n"
            )
        readme = (
            f"# {proof['title']}\n\n"
            "## Input\n\n"
            f"{chr(10).join(inputs)}\n\n"
            f"![input]({input_sheet})\n\n"
            "## Request\n\n"
            f"- task: `{proof['task_type']}`\n"
            f"- prompt: {prompt}\n\n"
            "## Expected result\n\n"
            f"{expected_lines}\n\n"
            "## Actual result\n\n"
            f"{observed_lines}\n\n"
            "## Run parameters\n\n"
            f"- seed: `{proof['seed']}`\n"
            f"- steps: `{proof['steps']}`\n"
            f"- guidance mode: `{proof['guidance_mode']}`\n"
            f"- output: `{proof['width']}x{proof['height']}`\n"
            f"- frames: `{proof['frames']}`\n"
            f"- fps: `{proof['fps']}`\n\n"
            f"{reproduce_section}"
            "## Official reference output\n\n"
            f"![official]({official_sheet})\n\n"
            "## mlx-gen output\n\n"
            f"![mlx-gen]({mlx_sheet})\n\n"
            "## Artifacts\n\n"
            f"- output: [{output_name}](./{output_name})\n"
            f"- metadata: [{metadata_name}](./{metadata_name})\n"
            "- proof: [proof.json](./proof.json)\n"
            f"- case json: `{cls._display_path(proof['case_json'])}`\n"
            f"- official output: `{cls._display_path(proof['official_output'])}`\n"
        )
        (case_out_dir / "README.md").write_text(readme)

    @staticmethod
    def _inspect_media(path: Path) -> dict:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            image = Image.open(path)
            return {
                "kind": "image",
                "width": int(image.width),
                "height": int(image.height),
                "frames": 1,
            }
        with av.open(str(path)) as container:
            stream = next(candidate for candidate in container.streams if candidate.type == "video")
            return {
                "kind": "video",
                "width": int(stream.width),
                "height": int(stream.height),
                "frames": int(stream.frames),
            }

    @classmethod
    def _input_items(
        cls,
        *,
        image_path: Path | None,
        video_path: Path | None,
        reference_video_paths: list[Path],
        reference_image_paths: list[Path],
    ) -> list[tuple[str, Image.Image]]:
        items: list[tuple[str, Image.Image]] = []
        if image_path is not None:
            items.append((f"source image: {image_path.name}", Image.open(image_path).convert("RGB")))
        for path in reference_image_paths:
            items.append((path.name, Image.open(path).convert("RGB")))
        if video_path is not None:
            for index in cls._video_preview_indices(video_path, count=3):
                items.append((f"source video f{index}", cls._read_video_frame(video_path, index)))
        for ref_index, path in enumerate(reference_video_paths):
            for index in cls._video_preview_indices(path, count=3):
                items.append((f"reference video {ref_index + 1} f{index}", cls._read_video_frame(path, index)))
        return items

    @classmethod
    def _output_items(cls, *, media_path: Path, media_spec: dict) -> list[tuple[str, Image.Image]]:
        if media_spec["kind"] == "image":
            return [(media_path.name, Image.open(media_path).convert("RGB"))]
        indices = cls._sample_frame_indices(int(media_spec["frames"]))
        return [(f"frame {index}", cls._read_video_frame(media_path, index)) for index in indices]

    @staticmethod
    def _sample_frame_indices(frame_count: int) -> list[int]:
        if frame_count <= 1:
            return [0]
        if frame_count <= 5:
            return list(range(frame_count))
        return sorted({0, frame_count // 4, frame_count // 2, (3 * frame_count) // 4, frame_count - 1})

    @classmethod
    def _video_preview_indices(cls, video_path: Path, *, count: int) -> list[int]:
        with av.open(str(video_path)) as container:
            stream = next(candidate for candidate in container.streams if candidate.type == "video")
            frame_count = int(stream.frames)
        if frame_count <= count:
            return list(range(frame_count))
        return [int(round(value)) for value in cls._linspace(0, frame_count - 1, count)]

    @staticmethod
    def _linspace(start: int, stop: int, count: int) -> list[float]:
        if count == 1:
            return [float(start)]
        return [float(start) + (float(stop) - float(start)) * index / float(count - 1) for index in range(count)]

    @staticmethod
    def _read_video_frame(video_path: Path, frame_index: int) -> Image.Image:
        with av.open(str(video_path)) as container:
            stream = next(candidate for candidate in container.streams if candidate.type == "video")
            for index, frame in enumerate(container.decode(stream)):
                if index == frame_index:
                    return frame.to_image().convert("RGB")
        raise ValueError(f"Frame {frame_index} not found in {video_path}")

    @classmethod
    def _build_sheet(
        cls,
        *,
        items: list[tuple[str, Image.Image]],
        title: str,
        columns: int,
    ) -> Image.Image:
        font = ImageFont.load_default()
        resized: list[tuple[str, Image.Image]] = []
        for label, image in items:
            if image.height > cls.MAX_CELL_HEIGHT:
                scale = cls.MAX_CELL_HEIGHT / float(image.height)
                image = image.resize((int(round(image.width * scale)), cls.MAX_CELL_HEIGHT), Image.Resampling.LANCZOS)
            resized.append((label, image))
        rows = max(1, math.ceil(len(resized) / columns))
        col_widths = [0] * columns
        row_heights = [0] * rows
        for index, (_, image) in enumerate(resized):
            row = index // columns
            col = index % columns
            col_widths[col] = max(col_widths[col], image.width)
            row_heights[row] = max(row_heights[row], image.height)
        width = cls.PAD * 2 + sum(col_widths) + cls.GAP * (columns - 1)
        height = cls.PAD * 2 + cls.TITLE_H + sum(row_heights) + rows * cls.LABEL_H + cls.GAP * max(0, rows - 1)
        sheet = Image.new("RGB", (width, height), cls.BG)
        draw = ImageDraw.Draw(sheet)
        draw.text((cls.PAD, cls.PAD), title, fill=cls.FG, font=font)
        y = cls.PAD + cls.TITLE_H
        item_index = 0
        for row in range(rows):
            x = cls.PAD
            for col in range(columns):
                if item_index >= len(resized):
                    break
                label, image = resized[item_index]
                draw.text((x, y), label, fill=cls.FG, font=font)
                sheet.paste(image, (x, y + cls.LABEL_H))
                x += col_widths[col] + cls.GAP
                item_index += 1
            y += row_heights[row] + cls.LABEL_H + cls.GAP
        return sheet


if __name__ == "__main__":
    BerniniOfficialPublicBundle.main()
