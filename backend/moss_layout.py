"""MOSS-Image layout prompts: the official draft → prompt conversion and the stage transforms around it.

A MOSS layout is ordinary prompt text — one line of JSON with a ``summary`` and a flat ``items`` list whose
boxes are absolute pixels around the canvas centre (``abs_centralized``, ``[y1, x1, y2, x2]``). The model is
never told the canvas size, so a box only means something at the canvas it was written for.

The user edits the official readable *draft* (``width``/``height``, top-left ``[x1, y1, x2, y2]``), exactly
as ``skills/moss-image-prompt-enhancer/scripts/compile_prompt.py`` (OpenMOSS/MOSS-Image @ c1355e6b) defines
it, and :func:`compile_draft` reproduces that tool byte for byte: same validation, same enclosure of
fractional edges, same compact serialisation, keys in the draft's own order. Around it Xiria adds what the
tool never needed, because it only ever targets one canvas:

- :func:`scale_draft` re-expresses a draft on another canvas (Hires redraws, a changed canvas size);
- :func:`region_draft` re-expresses it on an ADetailer crop, keeping the items that belong to the crop;
- ``prefix`` — text the workspace puts in front of the prompt (LoRA-group prompt, active presets) — joins the
  JSON on its own line.

The label set adds ``panel`` (used by the official examples) to the skill's three. Text is refused when it
holds lone surrogates or control characters other than newline and tab, so the browser's ``JSON.stringify``
mirror (``src/moss-layout.js``) serialises every accepted draft to the same bytes.
"""

from __future__ import annotations

import json
import math

LAYOUT_VERSION = 1
MAX_ITEMS = 256
MAX_PREFIX_CHARACTERS = 4000
MAX_TEXT_CHARACTERS = 4000
LABELS = ("object", "character", "text", "panel")
CANVAS_KINDS = ("single", "multi_panel", "collage")
ROOT_FIELDS = frozenset({"width", "height", "bbox_coord_space", "bbox_format", "summary", "items"})
SUMMARY_FIELDS = frozenset({"caption", "meta", "canvas", "palette"})
ITEM_FIELDS = frozenset({"label", "caption", "bbox", "text", "tags", "palette"})
ITEM_REQUIRED = frozenset({"label", "caption", "bbox"})
# An ADetailer crop keeps an item when this share of the item lies inside it, or when the item covers
# this share of the crop (a character box around a face crop, a full-canvas panel).
REGION_KEEP_RATIO = 0.25
# A stage whose aspect ratio differs from the draft's by more than this is reported: the boxes stretch.
ASPECT_WARNING_RATIO = 0.02


class LayoutError(ValueError):
    """A draft the compiler refuses; ``path`` names the field (``items[2].bbox``)."""

    def __init__(self, path: str, message: str):
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path
        self.message = message


def _text_problem(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "must be nonempty text"
    if len(value) > MAX_TEXT_CHARACTERS:
        return f"must be at most {MAX_TEXT_CHARACTERS} characters"
    for character in value:
        code = ord(character)
        if 0xD800 <= code <= 0xDFFF:
            return "contains an unpaired surrogate"
        if code < 0x20 and character not in "\n\t":
            return "contains a control character (only line breaks and tabs are allowed)"
    return None


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _enclose(x1, y1, x2, y2, width: int, height: int) -> list[int]:
    """The official conversion: enclose in pixel edges, then subtract the centre, in ``yxyx`` order."""
    return [
        math.floor(round(y1, 9)) - height // 2,
        math.floor(round(x1, 9)) - width // 2,
        math.ceil(round(y2, 9)) - height // 2,
        math.ceil(round(x2, 9)) - width // 2,
    ]


def draft_issues(draft) -> list[dict]:
    """Every problem in a draft, as ``{"path", "message"}``; empty when :func:`compile_draft` accepts it."""
    issues: list[dict] = []

    def add(path, message):
        issues.append({"path": path, "message": message})

    if not isinstance(draft, dict):
        add("", "the layout draft must be an object")
        return issues
    unknown = sorted(set(draft) - ROOT_FIELDS)
    missing = sorted(ROOT_FIELDS - set(draft))
    if unknown:
        add("", f"unknown fields {unknown}")
    if missing:
        add("", f"missing fields {missing}")
    width, height = draft.get("width"), draft.get("height")
    canvas_ok = True
    for name, value in (("width", width), ("height", height)):
        if name in draft and not (type(value) is int and value > 0 and value % 16 == 0):
            add(name, "must be a positive integer multiple of 16")
            canvas_ok = False
    if not ("width" in draft and "height" in draft):
        canvas_ok = False
    if "bbox_coord_space" in draft and draft["bbox_coord_space"] != "abs":
        add("bbox_coord_space", 'must be "abs" (top-left pixels)')
    if "bbox_format" in draft and draft["bbox_format"] != "xyxy":
        add("bbox_format", 'must be "xyxy"')

    summary = draft.get("summary")
    if "summary" in draft:
        if not isinstance(summary, dict):
            add("summary", "must be an object")
        else:
            unknown = sorted(set(summary) - SUMMARY_FIELDS)
            if unknown:
                add("summary", f"unknown fields {unknown}")
            if "caption" not in summary:
                add("summary.caption", "is required")
            else:
                problem = _text_problem(summary["caption"])
                if problem:
                    add("summary.caption", problem)
            if "canvas" in summary and summary["canvas"] not in CANVAS_KINDS:
                add("summary.canvas", "must be single, multi_panel or collage")
            for key in ("meta", "palette"):
                if key in summary:
                    _string_list_issues(summary[key], f"summary.{key}", add)

    items = draft.get("items")
    if "items" in draft:
        if not isinstance(items, list) or not items:
            add("items", "must be a nonempty array")
        elif len(items) > MAX_ITEMS:
            add("items", f"accepts at most {MAX_ITEMS} items")
        else:
            for index, item in enumerate(items):
                _item_issues(item, f"items[{index}]", width if canvas_ok else None, height if canvas_ok else None, add)
    return issues


def _string_list_issues(value, path, add):
    if not isinstance(value, list):
        add(path, "must be an array")
        return
    for index, entry in enumerate(value):
        problem = _text_problem(entry)
        if problem:
            add(f"{path}[{index}]", problem)


def _item_issues(item, path, width, height, add):
    if not isinstance(item, dict):
        add(path, "must be an object")
        return
    unknown = sorted(set(item) - ITEM_FIELDS)
    missing = sorted(ITEM_REQUIRED - set(item))
    if unknown:
        add(path, f"unknown fields {unknown}")
    if missing:
        add(path, f"missing fields {missing}")
    if "label" in item and item["label"] not in LABELS:
        add(f"{path}.label", f"must be one of {', '.join(LABELS)} (got {item['label']!r})")
    if "caption" in item:
        problem = _text_problem(item["caption"])
        if problem:
            add(f"{path}.caption", problem)
    if item.get("label") == "text" and "text" not in item:
        add(f"{path}.text", "text items require the literal text to render")
    if "text" in item:
        problem = _text_problem(item["text"])
        if problem:
            add(f"{path}.text", problem)
    for key in ("tags", "palette"):
        if key in item:
            _string_list_issues(item[key], f"{path}.{key}", add)
    if "bbox" in item:
        problem = _box_problem(item["bbox"], width, height)
        if problem:
            add(f"{path}.bbox", problem)


def _box_problem(box, width, height) -> str | None:
    if not isinstance(box, list) or len(box) != 4:
        return "must be [x1, y1, x2, y2]"
    if not all(_number(value) for value in box):
        return "must contain four finite numbers"
    x1, y1, x2, y2 = box
    if not (x1 < x2 and y1 < y2):
        return "must have positive width and height"
    if width is None or height is None:
        return None
    if not (0 <= x1 and x2 <= width and 0 <= y1 and y2 <= height):
        return f"exceeds the {width}x{height} canvas"
    result = _enclose(x1, y1, x2, y2, width, height)
    if not (result[0] < result[2] and result[1] < result[3]):
        return "is too small at export precision"
    return None


def require_valid_draft(draft) -> None:
    issues = draft_issues(draft)
    if issues:
        raise LayoutError(issues[0]["path"], issues[0]["message"])


def layout_json(draft) -> str:
    """The model-facing JSON for a valid draft — what the official tool prints for it."""
    require_valid_draft(draft)
    width, height = draft["width"], draft["height"]
    items = [
        {key: _enclose(*value, width, height) if key == "bbox" else value for key, value in item.items()}
        for item in draft["items"]
    ]
    result = {"summary": draft["summary"], "items": items,
              "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def join_prefix(prefix: str, body: str) -> str:
    prefix = (prefix or "").strip()
    return f"{prefix}\n{body}" if prefix else body


def compile_draft(draft, prefix: str = "") -> str:
    """The prompt for a draft at its own canvas: the official layout JSON, after the prefix line if any."""
    return join_prefix(prefix, layout_json(draft))


def _clamp(value, low, high):
    return min(high, max(low, value))


def scale_draft(draft, width: int, height: int) -> dict:
    """The same draft expressed on a ``width × height`` canvas (boxes scaled per axis, top-left space).

    Multiplying before dividing keeps the exact edges exact (``768 · 1024 / 768`` is 1024, where
    ``768 · (1024 / 768)`` is not), so a box on the canvas edge stays on it.
    """
    require_valid_draft(draft)
    source_width, source_height = draft["width"], draft["height"]
    if (source_width, source_height) == (width, height):
        return draft

    def scaled(box):
        x1, y1, x2, y2 = box
        return [
            _clamp(x1 * width / source_width, 0, width),
            _clamp(y1 * height / source_height, 0, height),
            _clamp(x2 * width / source_width, 0, width),
            _clamp(y2 * height / source_height, 0, height),
        ]

    items = [
        {key: scaled(value) if key == "bbox" else value for key, value in item.items()}
        for item in draft["items"]
    ]
    return _with_canvas(draft, width, height, items)


def _with_canvas(draft, width, height, items) -> dict:
    """A copy of ``draft`` on another canvas, its keys in the original order."""
    replaced = {"width": width, "height": height, "items": items}
    return {key: replaced.get(key, value) for key, value in draft.items()}


def aspect_mismatch(draft, width: int, height: int) -> bool:
    source = draft["width"] / draft["height"]
    return abs((width / height) / source - 1) > ASPECT_WARNING_RATIO


def region_draft(draft, image_size, crop_box, process_size) -> dict | None:
    """The draft as an ADetailer crop sees it, or ``None`` when no item belongs to the crop.

    The draft is first placed on the image the detector ran on (it may have been through Hires), then each
    item is intersected with the crop. An item is kept when at least ``REGION_KEEP_RATIO`` of it lies in the
    crop or it covers that share of the crop; the kept part is moved to the crop origin and scaled to the
    processing canvas. The summary is kept unchanged.
    """
    image_width, image_height = (int(value) for value in image_size)
    left, top, right, bottom = (float(value) for value in crop_box)
    process_width, process_height = (int(value) for value in process_size)
    if right <= left or bottom <= top:
        raise ValueError("ADetailer crop box must have a positive size")
    placed = scale_draft(draft, image_width, image_height)
    crop_area = (right - left) * (bottom - top)
    scale_x = process_width / (right - left)
    scale_y = process_height / (bottom - top)
    kept = []
    for item in placed["items"]:
        x1, y1, x2, y2 = item["bbox"]
        ix1, iy1, ix2, iy2 = max(x1, left), max(y1, top), min(x2, right), min(y2, bottom)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inside = (ix2 - ix1) * (iy2 - iy1)
        if inside < REGION_KEEP_RATIO * (x2 - x1) * (y2 - y1) and inside < REGION_KEEP_RATIO * crop_area:
            continue
        box = [
            _clamp((ix1 - left) * scale_x, 0, process_width),
            _clamp((iy1 - top) * scale_y, 0, process_height),
            _clamp((ix2 - left) * scale_x, 0, process_width),
            _clamp((iy2 - top) * scale_y, 0, process_height),
        ]
        enclosed = _enclose(*box, process_width, process_height)
        if not (enclosed[0] < enclosed[2] and enclosed[1] < enclosed[3]):
            continue
        kept.append({key: box if key == "bbox" else value for key, value in item.items()})
    if not kept:
        return None
    return _with_canvas(placed, process_width, process_height, kept)


def region_prompt(draft, prefix: str, image_size, crop_box, process_size) -> str:
    """The prompt an ADetailer region encodes: its crop-local layout, or the summary caption alone."""
    region = region_draft(draft, image_size, crop_box, process_size)
    if region is None:
        return join_prefix(prefix, draft["summary"]["caption"])
    return compile_draft(region, prefix)


def item_count(draft) -> int:
    return len(draft.get("items") or []) if isinstance(draft, dict) else 0


__all__ = [
    "ASPECT_WARNING_RATIO", "CANVAS_KINDS", "LABELS", "LAYOUT_VERSION", "LayoutError", "MAX_ITEMS",
    "MAX_PREFIX_CHARACTERS", "REGION_KEEP_RATIO", "aspect_mismatch", "compile_draft", "draft_issues",
    "item_count", "join_prefix", "layout_json", "region_draft", "region_prompt", "require_valid_draft",
    "scale_draft",
]
