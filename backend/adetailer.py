import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    confidence: float
    class_name: str
    mask: Image.Image | None = None


def discover_detector_models(root: Path):
    if not root.is_dir():
        return []
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.pt")
        if path.is_file()
    )


def resolve_detector_model(root: Path, requested: str):
    relative = Path(requested)
    if relative.is_absolute():
        raise ValueError("ADetailer detector path must be relative")
    try:
        path = (root / relative).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("ADetailer detector model does not exist") from error
    if not path.is_file() or root not in path.parents or path.suffix.lower() != ".pt":
        raise ValueError("ADetailer detector is outside the configured model directory")
    return path


def run_detector(python: Path, script: Path, model: Path, image: Image.Image, confidence: float):
    if not python.is_file():
        raise RuntimeError("Configured ADetailer Python executable is unavailable")
    if not script.is_file():
        raise RuntimeError("ADetailer detector worker is unavailable")

    with tempfile.TemporaryDirectory(prefix="xiriai-adetailer-") as temporary:
        temporary_root = Path(temporary)
        image_path = temporary_root / "input.png"
        mask_root = temporary_root / "masks"
        mask_root.mkdir()
        image.convert("RGB").save(image_path)
        payload = {
            "model": str(model),
            "image": str(image_path),
            "mask_directory": str(mask_root),
            "confidence": confidence,
        }
        environment = os.environ.copy()
        environment.setdefault("YOLO_CONFIG_DIR", str(temporary_root / "yolo-config"))
        result = subprocess.run(
            [str(python), str(script)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            env=environment,
            check=False,
        )
        if result.returncode != 0:
            message = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "Detector process failed"
            raise RuntimeError(f"ADetailer detection failed: {message}")
        try:
            output = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise RuntimeError("ADetailer detector returned invalid output") from error

        detections = []
        for item in output.get("detections", []):
            box = tuple(int(round(value)) for value in item["box"])
            mask = None
            if item.get("mask"):
                mask_path = (mask_root / item["mask"]).resolve(strict=True)
                if mask_root not in mask_path.parents:
                    raise RuntimeError("ADetailer detector returned an invalid mask path")
                with Image.open(mask_path) as opened:
                    mask = opened.convert("L").copy()
            detections.append(Detection(
                box=box,
                confidence=float(item["confidence"]),
                class_name=str(item.get("class_name") or "object"),
                mask=mask,
            ))
        return detections


def select_detections(detections, image_size, minimum_ratio, maximum_ratio, maximum_count):
    width, height = image_size
    image_area = width * height
    selected = []
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        clipped = (
            max(0, min(width, x1)),
            max(0, min(height, y1)),
            max(0, min(width, x2)),
            max(0, min(height, y2)),
        )
        area = max(0, clipped[2] - clipped[0]) * max(0, clipped[3] - clipped[1])
        ratio = area / image_area
        if area and minimum_ratio <= ratio <= maximum_ratio:
            detection.box = clipped
            selected.append(detection)
    selected.sort(key=lambda item: (item.box[2] - item.box[0]) * (item.box[3] - item.box[1]), reverse=True)
    return selected[:maximum_count] if maximum_count else selected


def detection_mask(detection: Detection, image_size, dilate_erode: int, blur: int):
    if detection.mask is not None:
        hard = detection.mask.resize(image_size, Image.Resampling.BILINEAR)
    else:
        hard = Image.new("L", image_size, 0)
        ImageDraw.Draw(hard).rectangle(detection.box, fill=255)

    array = np.asarray(hard) >= 128
    if dilate_erode:
        size = abs(dilate_erode)
        structure = np.ones((size, size), dtype=bool)
        operation = ndimage.binary_dilation if dilate_erode > 0 else ndimage.binary_erosion
        array = operation(array, structure=structure)
    hard = Image.fromarray(np.where(array, 255, 0).astype(np.uint8), mode="L")
    soft = hard.filter(ImageFilter.GaussianBlur(radius=blur)) if blur else hard.copy()
    return hard, soft


def render_detection_preview(image: Image.Image, detections: list[Detection]):
    canvas = image.convert("RGBA")
    palette = [
        (214, 255, 63),
        (0, 210, 255),
        (255, 117, 72),
        (181, 124, 255),
        (255, 205, 64),
    ]
    line_width = max(2, round(min(image.size) / 256))
    font_size = max(12, round(min(image.size) / 48))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    for detection in detections:
        color = palette[sum(ord(character) for character in detection.class_name) % len(palette)]
        x1, y1, x2, y2 = detection.box
        draw.rectangle((x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)), outline=(8, 9, 8, 255), width=line_width + 2)
        draw.rectangle((x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)), outline=(*color, 255), width=line_width)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
        label_width = text_box[2] - text_box[0] + line_width * 4
        label_height = text_box[3] - text_box[1] + line_width * 3
        label_y = y1 - label_height if y1 >= label_height else y1
        draw.rectangle((x1, label_y, min(image.width, x1 + label_width), min(image.height, label_y + label_height)), fill=(*color, 235))
        draw.text((x1 + line_width * 2, label_y + line_width), label, fill=(10, 11, 9, 255), font=font, stroke_width=1, stroke_fill=(*color, 255))
    return canvas.convert("RGB")


def expand_prompt(detail_prompt: str, parent_prompt: str):
    if not detail_prompt.strip():
        return parent_prompt
    return detail_prompt.replace("[PROMPT]", parent_prompt).strip()
