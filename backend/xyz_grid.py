"""QPlan2/P1 Phase D: whole-experiment grid layout, high-quality downscale and single-PNG export.

No torch and no inference globals. The caller supplies an experiment snapshot (axes labels,
per-cell status, frozen request and source asset path) and options; this module lays the grid out,
budgets it, renders it from the ORIGINAL server-side PNGs and publishes one PNG plus a JSON
manifest atomically, never overwriting a file.

Quality contract (plan §7.3): `original` / 100 % places every source 1:1 with no resampling or
colour conversion. A smaller percentage resizes each source ONCE, straight to its final rectangle,
with Pillow's Lanczos (a = 3) on float linear-light sRGB values, alpha premultiplied and safely
unpremultiplied; nothing ever allocates the native-resolution grid. Text and rules are drawn at the
target size with the bundled Noto Sans SC.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont, PngImagePlugin

FONT_PATH = Path(__file__).resolve().parent / "resources" / "fonts" / "NotoSansSC-Regular.otf"
FONT_SHA256 = "faa6c9df652116dde789d351359f3d7e5d2285a2b2a1f04a2d7244df706d5ea9"
FONT_DESCRIPTION = "Noto Sans SC Regular 2.004 (SIL OFL 1.1, bundled)"
SCHEMA_VERSION = 1
# Budgets are checked BEFORE anything is allocated. Refusal never changes scale or pages.
EXPORT_MAX_PIXELS = 200_000_000
EXPORT_MAX_EDGE = 60_000
EXPORT_MEMORY_BUDGET = 3 * 1024 ** 3
PNG_METADATA_LIMIT = 256 * 1024
MIN_READABLE_FONT = 10
LABEL_MAX_LINES = 6
ALGORITHM = "Pillow Image.Resampling.LANCZOS (Lanczos3), one pass per source, reducing_gap=None"
COLORSPACE = "linear-light: unprofiled 8-bit RGB(A) sources are sRGB by this project's convention; sRGB-decoded to float32, filtered, sRGB-encoded"
ALPHA_POLICY = "premultiplied before filtering; unpremultiplied by the float alpha where the stored 8-bit alpha > 0, RGB = 0 where it is 0; overshoot clipped"
STATUS_TEXT = {"pending": "未完成", "running": "生成中", "error": "失败", "skipped": "已跳过", "cancelled": "已取消", "missing": "原图缺失"}
THEMES = {
    "light": {"bg": (255, 255, 255), "header": (244, 245, 247), "text": (26, 27, 31), "muted": (98, 100, 108), "rule": (214, 216, 222),
              "placeholder": (236, 237, 241), "error": (168, 40, 40)},
    "dark": {"bg": (17, 17, 19), "header": (28, 29, 33), "text": (232, 232, 236), "muted": (160, 162, 170), "rule": (60, 62, 68),
             "placeholder": (34, 35, 40), "error": (238, 136, 136)},
}


# Summary text for LoRA axes: the export names the LoRAs, never the internal field ("X：lora_weight").
LORA_AXIS_TEXT = {"lora_weight": "LoRA 权重", "lora_model": "LoRA 替换", "lora_profile": "LoRA 组合", "lora_multiplier": "LoRA 组合倍率"}
AXIS_NAME_LIMIT = 4


def lora_display_name(ref):
    """The library name, or the file name without folders and extension."""
    name = str((ref or {}).get("name") or (ref or {}).get("value") or "").strip()
    name = re.split(r"[\\/]", name)[-1]
    return re.sub(r"\.safetensors$", "", name, flags=re.IGNORECASE) or ""


def _names(items):
    names = list(dict.fromkeys(name for name in items if name))
    return "、".join(names[:AXIS_NAME_LIMIT]) + (f" 等 {len(names)} 个" if len(names) > AXIS_NAME_LIMIT else "")


def axis_text(key, axis, baseline_loras=()):
    """One axis for the export summary, identical for X, Y and Z. LoRA axes name their LoRAs: the weighted LoRA, the
    swapped LoRA and its candidates, or the LoRAs a set tests; any other axis keeps its field name."""
    field = (axis or {}).get("field")
    if field not in LORA_AXIS_TEXT:
        return f"{key.upper()}：{field}"
    by_id = {entry.get("instanceId"): entry for entry in baseline_loras or () if isinstance(entry, dict)}
    named = lambda instance_id, fallback=None: lora_display_name(by_id.get(instance_id) or fallback or {"name": instance_id or ""})
    values = [item.get("value") for item in axis.get("values") or [] if isinstance(item, dict)]
    if field == "lora_weight":
        detail = named(axis.get("target"))
    elif field == "lora_model":
        candidates = _names(lora_display_name(value) for value in values if isinstance(value, dict))
        detail = f"{named(axis.get('target'))} → {candidates}" if candidates else named(axis.get("target"))
    else:
        snapshots = [value for value in values if isinstance(value, dict)] if field == "lora_profile" else [axis.get("snapshot") or {}]
        names = []
        for snapshot in snapshots:
            members = {member.get("instanceId"): member for member in snapshot.get("members") or [] if isinstance(member, dict)}
            targets = [target for target in snapshot.get("test_targets") or [] if isinstance(target, str)]
            names += [named(target, members.get(target)) for target in targets] if targets else [lora_display_name(member) for member in members.values()]
        detail = _names(names)
    return f"{key.upper()}：{LORA_AXIS_TEXT[field]}" + (f" · {detail}" if detail else "")


class GridError(ValueError):
    def __init__(self, code, message, status=422, **extra):
        super().__init__(message)
        self.code = code
        self.status = status
        self.extra = extra


class ExportCancelled(Exception):
    pass


# Colour ------------------------------------------------------------------------------------------

_SRGB_TO_LINEAR = np.array([(value / 255) / 12.92 if value / 255 <= 0.04045 else ((value / 255 + 0.055) / 1.055) ** 2.4
                            for value in range(256)], dtype=np.float32)


def linear_to_srgb_u8(linear):
    linear = np.clip(linear, 0.0, 1.0)
    encoded = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    return np.floor(encoded * 255 + 0.5).astype(np.uint8)


_RESIZE_CALLS = [0]


def _resize_channel(channel, size):
    """ONE Lanczos3 pass from the source grid to the target grid (never a cascade)."""
    _RESIZE_CALLS[0] += 1
    return np.asarray(Image.fromarray(channel).resize(size, Image.Resampling.LANCZOS, reducing_gap=None), dtype=np.float32)


def downscale_linear(image, size):
    """Source → target in one Lanczos3 pass on linear-light, premultiplied values."""
    has_alpha = image.mode == "RGBA"
    array = np.asarray(image)
    linear = _SRGB_TO_LINEAR[array[..., :3]]
    if has_alpha:
        alpha = array[..., 3].astype(np.float32) / 255.0
        linear *= alpha[..., None]
    channels = [_resize_channel(np.ascontiguousarray(linear[..., index]), size) for index in range(3)]
    if has_alpha:
        alpha_out = np.clip(_resize_channel(np.ascontiguousarray(alpha), size), 0.0, 1.0)
        alpha_u8 = np.floor(alpha_out * 255 + 0.5).astype(np.uint8)
        # Visibility is decided by the stored 8-bit alpha: a pixel that rounds to alpha 0 keeps
        # RGB 0 (no hidden colour, no amplified residue); visible ones divide by the float alpha.
        safe = alpha_u8 > 0
        rgb = np.stack([np.where(safe, channel / np.where(safe, alpha_out, 1.0), 0.0) for channel in channels], axis=-1)
        out = np.concatenate([linear_to_srgb_u8(rgb), alpha_u8[..., None]], axis=-1)
        return Image.fromarray(out, "RGBA")
    return Image.fromarray(linear_to_srgb_u8(np.stack(channels, axis=-1)), "RGB")


def open_source(path, expectation=None, fingerprint=None):
    """Decode one ORIGINAL PNG asset into RGB or RGBA without any silent colour-space change."""
    with open(path, 'rb') as source:
        verification = None
        if expectation is not None:
            from .xyz_assets import verify_handle, AssetVerificationError
            try:
                verification = verify_handle(source, expectation, fingerprint)
            except AssetVerificationError as error:
                raise GridError('source-changed', str(error)) from error
        with Image.open(source) as handle:
            handle.load()
            info = dict(handle.info)
            image = handle.copy()
        if verification is not None:
            try:
                after = verify_handle(source, expectation, fingerprint)
                if (after['sha256'], after['bytes']) != (verification['sha256'], verification['bytes']):
                    raise AssetVerificationError('原图在解码期间发生变化')
            except AssetVerificationError as error:
                raise GridError('source-changed', str(error)) from error
            image.info['_xyz_source_sha256'] = verification['sha256']
    profile = info.get("icc_profile")
    if profile:
        try:
            description = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(profile))) or ""
        except (OSError, ImageCms.PyCMSError, TypeError) as error:
            raise GridError("icc", f"{Path(path).name}: 嵌入的色彩配置损坏，已拒绝导出（不做隐式转换）") from error
        if "srgb" not in description.lower():
            raise GridError("icc", f"{Path(path).name}: 嵌入了非 sRGB 色彩配置（{description.strip()[:60]}），已拒绝导出（不做隐式转换）")
    gamma = info.get("gamma")
    if gamma is not None and abs(float(gamma) - 1 / 2.2) > 0.01 and "srgb" not in info:
        raise GridError("gamma", f"{Path(path).name}: PNG gAMA={gamma} 不是 sRGB 约定，已拒绝导出")
    if image.mode in {"RGB", "RGBA"}:
        return image
    if image.mode in {"L", "P", "LA", "PA"}:
        # Lossless expansion of 8-bit grey/palette data.
        return image.convert("RGBA" if image.mode in {"LA", "PA"} or (image.mode == "P" and "transparency" in info) else "RGB")
    raise GridError("mode", f"{Path(path).name}: 像素格式 {image.mode} 不支持原样导出（仅 8 位 RGB/RGBA/灰度/调色板）")


# Text --------------------------------------------------------------------------------------------

_FONTS = {}


def font(size):
    size = max(1, int(size))
    if size not in _FONTS:
        _FONTS[size] = ImageFont.truetype(str(FONT_PATH), size)
    return _FONTS[size]


def font_sha256():
    return hashlib.sha256(FONT_PATH.read_bytes()).hexdigest()


def wrap(text, face, width, max_lines=LABEL_MAX_LINES):
    """Greedy wrap preferring spaces; CJK breaks between characters. Returns (lines, truncated)."""
    text = " ".join(str(text).split()) or "—"
    width = max(1, width)
    lines, current = [], ""
    for character in text:
        candidate = current + character
        if not current or face.getlength(candidate) <= width:
            current = candidate
            continue
        cut = current.rfind(" ")
        if cut > 0 and character != " ":
            lines.append(current[:cut])
            current = current[cut + 1:] + character
        else:
            lines.append(current.rstrip())
            current = character.lstrip()
    if current:
        lines.append(current)
    truncated = len(lines) > max_lines
    if truncated:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and face.getlength(last + "…") > width:
            last = last[:-1]
        lines[-1] = last + "…"
    return lines, truncated


def line_height(face):
    ascent, descent = face.getmetrics()
    return ascent + descent


# Layout ------------------------------------------------------------------------------------------

def _scaled(value, percent):
    return 0 if value <= 0 else max(1, (value * percent + 50) // 100)


def image_target_size(width, height, percent):
    """One factor for both dimensions, deterministic round-half-up integer rounding."""
    return max(1, (width * percent + 50) // 100), max(1, (height * percent + 50) // 100)


def _blocks(snapshot, options):
    zs = range(len(snapshot["axes"]["z"]["labels"]) or 1)
    samples = range(snapshot.get("repeat_count", 1) or 1)
    if options["scope"] == "page":
        page = options.get("page") or {}
        z, sample = page.get("z_index", 0), page.get("sample_index", 0)
        if z not in zs or sample not in samples:
            raise GridError("page", "所选 Z / 样本页不存在")
        return [(z, sample)]
    return [(z, sample) for z in zs for sample in samples]


def compute_layout(snapshot, options, percent=None):
    """Every rectangle and text line of the export at `percent` (100 for original)."""
    percent = 100 if options["mode"] == "original" else (percent or options["percent"])
    blocks = _blocks(snapshot, options)
    cells = snapshot["cells"]
    included = [cell for cell in cells if (cell["coordinate"][2], cell["coordinate"][3]) in set(blocks)]
    sizes = [(cell["width"], cell["height"]) for cell in included if cell["status"] == "complete" and cell.get("width")]
    if not sizes:
        raise GridError("no-complete", "所选范围没有任何已完成的格；至少需要 1 张已完成原图才能导出")
    base = sorted(min(w, h) for w, h in sizes)[len(sizes) // 2]
    native = {"font": min(56, max(14, round(base * 0.045)))}
    native.update(title=round(native["font"] * 1.25), caption=max(10, round(native["font"] * 0.75)),
                  gap=max(4, round(base * 0.016)), margin=max(16, round(base * 0.05)), pad=max(4, round(native["font"] * 0.4)),
                  rule=max(1, round(base / 1024)))
    s = {key: _scaled(value, percent) for key, value in native.items()}
    label_face, title_face, caption_face = font(s["font"]), font(s["title"]), font(s["caption"])
    label_line, title_line, caption_line = line_height(label_face), line_height(title_face), line_height(caption_face)
    axes = snapshot["axes"]
    columns, rows = max(1, len(axes["x"]["labels"])), max(1, len(axes["y"]["labels"]))
    median_target = image_target_size(*sorted(sizes)[len(sizes) // 2], percent)
    col_w, row_h = [0] * columns, [0] * rows
    for cell in included:
        x, y = cell["coordinate"][0], cell["coordinate"][1]
        if cell["status"] == "complete" and cell.get("width"):
            tw, th = image_target_size(cell["width"], cell["height"], percent)
            col_w[x], row_h[y] = max(col_w[x], tw), max(row_h[y], th)
    col_w = [width or median_target[0] for width in col_w]
    row_h = [height or median_target[1] for height in row_h]
    truncated = []
    use_x, use_y = bool(axes["x"].get("field")), bool(axes["y"].get("field"))
    x_lines = [wrap(axes["x"]["labels"][index] if use_x else "", label_face, col_w[index] - 2 * s["pad"]) for index in range(columns)]
    truncated += [axes["x"]["labels"][index] for index, (_lines, cut) in enumerate(x_lines) if cut]
    header_h = (max(len(lines) for lines, _cut in x_lines) * label_line + 2 * s["pad"]) if use_x else 0
    ylabel_w = 0
    y_lines = []
    if use_y:
        widest = max(label_face.getlength(" ".join(str(label).split()) or "—") for label in axes["y"]["labels"])
        ylabel_w = int(min(widest + 2 * s["pad"], max(col_w) * 0.6 + 2 * s["pad"], max(s["font"] * 14, 1)))
        y_lines = [wrap(axes["y"]["labels"][index], label_face, ylabel_w - 2 * s["pad"]) for index in range(rows)]
        truncated += [axes["y"]["labels"][index] for index, (_lines, cut) in enumerate(y_lines) if cut]
    # Optional per-cell tokens (a search grid's numbered LoRA weights) share ONE slot grid: the same slot width, the same
    # count per line and the same height in every cell, so values line up across the whole export.
    token_cells = [cell for cell in included if cell.get("tokens")]
    token_grid = None
    if token_cells:
        token_gap = caption_face.getlength("  ")
        token_w = min(max(caption_face.getlength(token) for cell in token_cells for token in cell["tokens"]), min(col_w))
        most = max(len(cell["tokens"]) for cell in token_cells)
        per_line = max(1, min(most, int((min(col_w) + token_gap) // (token_w + token_gap))))
        lines = math.ceil(most / per_line)
        token_grid = {"slot": token_w + token_gap, "width": token_w, "per_line": math.ceil(most / lines), "lines": lines}
    caption_h = (caption_line + s["pad"] if options["captions"] else 0) + (token_grid["lines"] * caption_line + s["pad"] if token_grid else 0)
    grid_w = ylabel_w + (s["gap"] if use_y else 0) + sum(col_w) + s["gap"] * (columns - 1)
    # An unused Z axis still carries the planner's "—" label: a heading only when Z is a real axis
    # or there are repeat samples to tell apart.
    use_z = bool(axes["z"].get("field"))
    show_heading = use_z or (snapshot.get("repeat_count", 1) or 1) > 1
    by_coordinate = {tuple(cell["coordinate"]): cell for cell in cells}
    block_ops = []
    for z, sample in blocks:
        heading = []
        if show_heading:
            text = []
            if use_z:
                text.append(f"Z{z + 1} · {axes['z']['labels'][z]}")
            if (snapshot.get("repeat_count", 1) or 1) > 1:
                labels = snapshot.get("sample_labels") or []
                text.append(labels[sample] if sample < len(labels) else f"样本 {sample + 1}")
            heading, cut = wrap(" · ".join(text), title_face, grid_w)
            if cut:
                truncated.append(" · ".join(text))
        heading_h = len(heading) * title_line + (s["pad"] if heading else 0)
        height = heading_h + header_h + sum(row_h) + caption_h * rows + s["gap"] * (rows - 1)
        block_ops.append({"z": z, "sample": sample, "heading": heading, "heading_h": heading_h, "w": grid_w, "h": height})
    arrangement = options["arrangement"]
    count = len(block_ops)
    per_row = count if arrangement == "horizontal" else 1 if arrangement == "vertical" else max(1, math.ceil(math.sqrt(count)))
    rows_of_blocks = math.ceil(count / per_row)
    block_gap = s["margin"]
    cell_w = max(op["w"] for op in block_ops)
    row_heights = [max(op["h"] for op in block_ops[r * per_row:(r + 1) * per_row]) for r in range(rows_of_blocks)]
    content_w = per_row * cell_w + block_gap * (per_row - 1) if count > 1 else cell_w
    title_text = snapshot["title"]
    title_lines, title_cut = wrap(title_text, title_face, content_w, max_lines=3)
    if title_cut:
        truncated.append(title_text)
    subtitle_lines, sub_cut = wrap(snapshot["subtitle"], label_face, content_w, max_lines=3)
    if sub_cut:
        truncated.append(snapshot["subtitle"])
    # Optional legend (a search grid's numbered test LoRAs): fixed-width columns, each entry on one line, then a note.
    legend = snapshot.get("legend") or {}
    legend_items, legend_note, legend_cols = legend.get("items") or [], [], 1
    legend_head = []
    if legend_items:
        legend_gap = label_face.getlength("    ")
        legend_widths = [min(label_face.getlength(item), content_w) for item in legend_items]
        legend_w = max(legend_widths)
        legend_flow = sum(legend_widths) + legend_gap * (len(legend_items) - 1) <= content_w
        # One line: entries flow with even gaps. Several lines: equal-width columns so entries line up row to row.
        legend_cols = len(legend_items) if legend_flow else max(1, min(len(legend_items), int((content_w + legend_gap) // (legend_w + legend_gap))))
        legend_cols = math.ceil(len(legend_items) / math.ceil(len(legend_items) / legend_cols))
        legend_head = [legend.get("heading") or "测试 LoRA（编号对应每格下方的权重）"]
        if legend.get("note"):
            legend_note, note_cut = wrap(legend["note"], label_face, content_w, max_lines=3)
            if note_cut:
                truncated.append(legend["note"])
    legend_rows = math.ceil(len(legend_items) / legend_cols) if legend_items else 0
    legend_h = (s["pad"] + (len(legend_head) + legend_rows + len(legend_note)) * label_line) if legend_items else 0
    title_h = len(title_lines) * title_line + len(subtitle_lines) * label_line + legend_h + s["pad"] * 2
    width = 2 * s["margin"] + content_w
    height = 2 * s["margin"] + title_h + block_gap + sum(row_heights) + block_gap * (rows_of_blocks - 1)
    ops = {"texts": [], "rects": [], "rules": [], "cells": []}
    y = s["margin"]
    for line in title_lines:
        ops["texts"].append({"xy": (s["margin"], y), "text": line, "size": s["title"], "color": "text"}); y += title_line
    for line in subtitle_lines:
        ops["texts"].append({"xy": (s["margin"], y), "text": line, "size": s["font"], "color": "muted"}); y += label_line
    if legend_items:
        y += s["pad"]
        for line in legend_head:
            ops["texts"].append({"xy": (s["margin"], y), "text": line, "size": s["font"], "color": "muted"}); y += label_line
        flow_x = s["margin"]
        for index, item in enumerate(legend_items):
            row, column = divmod(index, legend_cols)
            x = flow_x if legend_flow else s["margin"] + column * (legend_w + legend_gap)
            ops["texts"].append({"xy": (x, y + row * label_line), "text": wrap(item, label_face, legend_w, max_lines=1)[0][0], "size": s["font"], "color": "text"})
            flow_x += legend_widths[index] + legend_gap
        y += legend_rows * label_line
        for line in legend_note:
            ops["texts"].append({"xy": (s["margin"], y), "text": line, "size": s["font"], "color": "muted"}); y += label_line
    top = s["margin"] + title_h + block_gap
    for index, op in enumerate(block_ops):
        r, c = divmod(index, per_row)
        bx = s["margin"] + c * (cell_w + block_gap)
        by = top + sum(row_heights[:r]) + block_gap * r
        y = by
        for line in op["heading"]:
            ops["texts"].append({"xy": (bx, y), "text": line, "size": s["title"], "color": "text"}); y += title_line
        y = by + op["heading_h"]
        grid_x = bx + ylabel_w + (s["gap"] if use_y else 0)
        if use_x:
            ops["rects"].append({"box": (grid_x, y, sum(col_w) + s["gap"] * (columns - 1), header_h), "color": "header"})
            cx = grid_x
            for index, (lines, _cut) in enumerate(x_lines):
                ty = y + s["pad"]
                for line in lines:
                    lw = label_face.getlength(line)
                    ops["texts"].append({"xy": (cx + max(0, (col_w[index] - lw) / 2), ty), "text": line, "size": s["font"], "color": "text"}); ty += label_line
                cx += col_w[index] + s["gap"]
            ops["rules"].append({"box": (grid_x, y + header_h - s["rule"], sum(col_w) + s["gap"] * (columns - 1), s["rule"])})
            y += header_h
        cy = y
        for row in range(rows):
            if use_y:
                ops["rects"].append({"box": (bx, cy, ylabel_w, row_h[row] + caption_h), "color": "header"})
                lines = y_lines[row][0]
                ty = cy + max(0, (row_h[row] - len(lines) * label_line) // 2)
                for line in lines:
                    ops["texts"].append({"xy": (bx + s["pad"], ty), "text": line, "size": s["font"], "color": "text"}); ty += label_line
            cx = grid_x
            for column in range(columns):
                cell = by_coordinate.get((column, row, op["z"], op["sample"]))
                slot = (cx, cy, col_w[column], row_h[row])
                entry = {"cell": cell["id"] if cell else None, "slot": slot, "status": cell["status"] if cell else "missing", "image": None}
                if cell and cell["status"] == "complete" and cell.get("width"):
                    tw, th = image_target_size(cell["width"], cell["height"], percent)
                    entry["image"] = (cx + (col_w[column] - tw) // 2, cy + (row_h[row] - th) // 2, tw, th)
                if options["captions"]:
                    # A search round supplies its own persistent caption (candidate id, displacement, seed).
                    caption = (cell.get("caption") or (f"种子 {cell['seed']}" if entry["image"] else STATUS_TEXT.get(entry["status"], entry["status"]))) if cell else "—"
                    entry["caption"] = {"xy": (cx, cy + row_h[row] + max(1, s["pad"] // 2)), "text": wrap(caption, caption_face, col_w[column])[0][0], "size": s["caption"]}
                if token_grid and cell and cell.get("tokens"):
                    top_y = cy + row_h[row] + max(1, s["pad"] // 2) + (caption_line + s["pad"] // 2 if options["captions"] else 0)
                    entry["tokens"] = [{"xy": (cx + (i % token_grid["per_line"]) * token_grid["slot"], top_y + (i // token_grid["per_line"]) * caption_line),
                                        "text": wrap(token, caption_face, token_grid["width"], max_lines=1)[0][0], "size": s["caption"]}
                                       for i, token in enumerate(cell["tokens"])]
                ops["cells"].append(entry)
                cx += col_w[column] + s["gap"]
            cy += row_h[row] + caption_h + s["gap"]
        if index < count - 1 and per_row == 1:
            ops["rules"].append({"box": (bx, by + row_heights[r] + block_gap // 2, cell_w, s["rule"])})
    warnings = []
    if s["font"] < MIN_READABLE_FONT:
        warnings.append({"code": "tiny-labels", "message": f"标签字号在 {percent}% 下仅 {s['font']} px，可能难以辨认；请提高百分比（不会自动调整或省略标签）"})
    tiny = min(min(col_w), min(row_h))
    if tiny < 32:
        warnings.append({"code": "tiny-cells", "message": f"最小格在 {percent}% 下仅 {tiny} px；请提高百分比"})
    if truncated:
        warnings.append({"code": "long-labels", "message": f"{len(truncated)} 个标签超过 {LABEL_MAX_LINES} 行已截断显示；清单保留完整文本"})
    alpha = any(cell.get("alpha") for cell in included if cell["status"] == "complete")
    return {"percent": percent, "width": width, "height": height, "alpha": alpha, "blocks": [(op["z"], op["sample"]) for op in block_ops],
            "ops": ops, "warnings": warnings, "fonts": s, "columns": col_w, "rows": row_h, "truncated_labels": truncated}


def memory_estimate(layout, snapshot):
    channels = 4 if layout["alpha"] else 3
    canvas = layout["width"] * layout["height"] * channels
    largest = max((cell["width"] * cell["height"] for cell in snapshot["cells"] if cell["status"] == "complete" and cell.get("width")), default=0)
    # Decoded uint8 source + (scaled path) float32 linear copy, per-channel float images and target floats.
    workspace = largest * 4 + (largest * 4 * 4 * 3 if layout["percent"] < 100 else 0)
    encoder = canvas // 2
    return canvas + workspace + encoder


def check_budget(layout, snapshot):
    pixels = layout["width"] * layout["height"]
    memory = memory_estimate(layout, snapshot)
    problems = []
    if pixels > EXPORT_MAX_PIXELS:
        problems.append(f"像素 {pixels / 1e6:.1f} MP 超过上限 {EXPORT_MAX_PIXELS / 1e6:.0f} MP")
    if max(layout["width"], layout["height"]) > EXPORT_MAX_EDGE:
        problems.append(f"边长 {max(layout['width'], layout['height'])} px 超过上限 {EXPORT_MAX_EDGE} px")
    if memory > EXPORT_MEMORY_BUDGET:
        problems.append(f"预计内存 {memory / 1024 ** 3:.2f} GiB 超过预算 {EXPORT_MEMORY_BUDGET / 1024 ** 3:.0f} GiB")
    return {"ok": not problems, "pixels": pixels, "memory_bytes": memory, "problems": problems,
            "limits": {"max_pixels": EXPORT_MAX_PIXELS, "max_edge": EXPORT_MAX_EDGE, "memory_bytes": EXPORT_MEMORY_BUDGET}}


def estimate(snapshot, options):
    """Final dimensions, pixels, memory and budget verdict; suggested percentages when refused.
    Suggestions are offered for the user to choose — never applied."""
    layout = compute_layout(snapshot, options)
    budget = check_budget(layout, snapshot)
    suggestions = []
    if not budget["ok"]:
        low, high, best = 1, (layout["percent"] - 1), None
        while low <= high:
            middle = (low + high) // 2
            if check_budget(compute_layout(snapshot, {**options, "mode": "scaled"}, middle), snapshot)["ok"]:
                best, low = middle, middle + 1
            else:
                high = middle - 1
        if best:
            suggestions = sorted({best, max(1, best * 3 // 4), max(1, best // 2)}, reverse=True)
    counts = {}
    for cell in snapshot["cells"]:
        if (cell["coordinate"][2], cell["coordinate"][3]) in set(layout["blocks"]):
            counts[cell["status"]] = counts.get(cell["status"], 0) + 1
    return {"width": layout["width"], "height": layout["height"], "percent": layout["percent"], "alpha": layout["alpha"],
            "blocks": len(layout["blocks"]), "cells": counts, "budget": budget, "warnings": layout["warnings"], "suggestions": suggestions,
            "font_px": layout["fonts"]["font"]}


# Render and publish ------------------------------------------------------------------------------

def _source_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(snapshot, options, layout, *, cancelled=lambda: False, progress=lambda done, total: None):
    """Draw the chrome at target size, then place each ORIGINAL source sequentially."""
    theme = THEMES[options["theme"]]
    mode = "RGBA" if layout["alpha"] else "RGB"
    canvas = Image.new(mode, (layout["width"], layout["height"]), theme["bg"] + ((255,) if mode == "RGBA" else ()))
    draw = ImageDraw.Draw(canvas)
    for rect in layout["ops"]["rects"]:
        x, y, w, h = rect["box"]
        draw.rectangle((x, y, x + w - 1, y + h - 1), fill=theme[rect["color"]])
    for rule in layout["ops"]["rules"]:
        x, y, w, h = rule["box"]
        draw.rectangle((x, y, x + w - 1, y + h - 1), fill=theme["rule"])
    for text in layout["ops"]["texts"]:
        draw.text(text["xy"], text["text"], font=font(text["size"]), fill=theme[text["color"]])
    placed, sources = [], {cell["id"]: cell for cell in snapshot["cells"]}
    jobs = [entry for entry in layout["ops"]["cells"] if entry["image"]]
    for entry in layout["ops"]["cells"]:
        x, y, w, h = entry["slot"]
        if not entry["image"]:
            draw.rectangle((x, y, x + w - 1, y + h - 1), fill=theme["placeholder"], outline=theme["rule"], width=max(1, layout["fonts"]["rule"]))
            label = STATUS_TEXT.get(entry["status"], entry["status"])
            face = font(layout["fonts"]["font"])
            lw = face.getlength(label)
            draw.text((x + max(0, (w - lw) / 2), y + max(0, (h - line_height(face)) / 2)), label, font=face,
                      fill=theme["error"] if entry["status"] in {"error", "missing"} else theme["muted"])
        if entry.get("caption"):
            draw.text(entry["caption"]["xy"], entry["caption"]["text"], font=font(entry["caption"]["size"]), fill=theme["muted"])
        for token in entry.get("tokens") or ():
            draw.text(token["xy"], token["text"], font=font(token["size"]), fill=theme["text"])
    for done, entry in enumerate(jobs, start=1):
        if cancelled():
            raise ExportCancelled()
        cell = sources[entry["cell"]]
        image = open_source(cell["path"], cell.get('asset_expectation'), cell.get('asset_fingerprint'))
        source_sha = image.info.get('_xyz_source_sha256')
        if image.size != (cell["width"], cell["height"]):
            raise GridError("source-changed", f"格 {cell['id']} 的原图尺寸已变化（{image.size} ≠ 记录的 {cell['width']}×{cell['height']}）；未导出")
        x, y, w, h = entry["image"]
        if layout["percent"] < 100:
            image = downscale_linear(image, (w, h))
        if image.mode != mode:
            image = image.convert(mode)  # RGB → RGBA adds opaque alpha; lossless
        canvas.paste(image, (x, y))  # replace pixels (alpha included), never composite
        placed.append({"cell": cell["id"], "rect": [x, y, w, h], "source_size": [cell["width"], cell["height"]],
                       'source_sha256': source_sha})
        progress(done, len(jobs))
    return canvas, placed


def build_manifest(snapshot, options, layout, placed, export_id, source_hashes):
    placed_by_cell = {item["cell"]: item for item in placed}
    return {
        "schema_version": SCHEMA_VERSION, "kind": "xyz_plot_grid_manifest", "export_id": export_id,
        "experiment": {"id": snapshot["experiment_id"], "definition_digest": snapshot["digest"], "engine": snapshot["engine"],
                       "axes": snapshot["axes"], "dimensions": snapshot["dimensions"], "repeat_count": snapshot.get("repeat_count", 1),
                       "baseline": snapshot["baseline"], "title": snapshot["title"], "subtitle": snapshot["subtitle"]},
        "recipe": recipe(options, layout),
        # QPlan2/P2: a search grid records its rules, mapping and global/test split, not the first cell's parameters.
        **({"search": snapshot["search"]} if snapshot.get("search") else {}),
        "warnings": layout["warnings"], "truncated_labels": layout["truncated_labels"],
        "cells": [{
            "id": cell["id"], "coordinate": cell["coordinate"], "labels": cell["labels"], "status": cell["status"], "seed": cell.get("seed"),
            **({"candidate": cell["candidate"], "origin_coordinate": cell.get("origin_coordinate")} if cell.get("candidate") else {}),
            "request": cell.get("request"), "lora_references": cell.get("lora_references"), "batch": cell.get("batch"),
            "asset": ({"relative": cell["relative"], "asset_id": cell.get("asset_id"), "width": cell["width"], "height": cell["height"],
                       "sha256": source_hashes.get(cell["id"])} if cell.get("relative") else None),
            "placed": placed_by_cell.get(cell["id"]),
        } for cell in snapshot["cells"] if (cell["coordinate"][2], cell["coordinate"][3]) in set(layout["blocks"])],
    }


def recipe(options, layout):
    return {"scope": options["scope"], "page": options.get("page"), "mode": options["mode"], "percent": layout["percent"],
            "algorithm": None if layout["percent"] == 100 else ALGORITHM,
            "colorspace": "sources placed 1:1, no colour conversion" if layout["percent"] == 100 else COLORSPACE,
            "alpha": "preserved 1:1" if layout["percent"] == 100 else ALPHA_POLICY,
            "rounding": "target = floor(source × percent / 100 + 0.5), min 1, same factor for both axes",
            "theme": options["theme"], "arrangement": options["arrangement"], "captions": options["captions"],
            **({"lora_weights": options["lora_weights"]} if "lora_weights" in options else {}),
            "font": {"description": FONT_DESCRIPTION, "sha256": FONT_SHA256, "label_px": layout["fonts"]["font"]},
            "canvas": {"width": layout["width"], "height": layout["height"], "mode": "RGBA" if layout["alpha"] else "RGB"},
            "blocks": [list(block) for block in layout["blocks"]], "columns": layout["columns"], "rows": layout["rows"]}


def png_metadata(snapshot, layout, options, export_id, manifest_sha256, placed):
    """Independent grid record; never a cell's `parameters` or ComfyUI `workflow`."""
    record = {"schema_version": SCHEMA_VERSION, "kind": "xyz_plot_grid", "export_id": export_id,
              "experiment_id": snapshot["experiment_id"], "definition_digest": snapshot["digest"],
              "manifest": {"name": "same stem, .json", "sha256": manifest_sha256}, "recipe": recipe(options, layout),
              **({"search": {key: snapshot["search"][key] for key in ("session_id", "round", "context_hash", "columns", "rows", "candidates_count")}} if snapshot.get("search") else {}),
              "cells": [{"id": item["cell"], "rect": item["rect"]} for item in placed]}
    text = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    overflow = len(text.encode("utf-8")) > PNG_METADATA_LIMIT
    if overflow:
        record.pop("cells")
        record["metadata_overflow"] = {"limit_bytes": PNG_METADATA_LIMIT, "omitted": "per-cell rectangles", "see": "manifest"}
        text = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    info = PngImagePlugin.PngInfo()
    info.add_itxt("xyz_plot", text)
    return info, overflow


def _publish_sibling(temporary, destination):
    """Exclusive create of a sibling file (manifest); never replaces an existing file."""
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise
    except OSError:
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(descriptor)
        os.replace(temporary, destination)
        return
    os.unlink(temporary)


def export_grid(snapshot, options, *, directory, stem, publish, export_id, cancelled=lambda: False, progress=lambda done, total: None):
    """Budget → render → manifest → PNG with metadata → atomic exclusive publication.

    `publish(temporary_png) -> final_png_path` is the caller's exclusive publisher (never overwrites).
    Temporary files are removed on every failure or cancellation; completed originals are only read.
    """
    layout = compute_layout(snapshot, options)
    budget = check_budget(layout, snapshot)
    if not budget["ok"]:
        raise GridError("budget", "；".join(budget["problems"]), suggestions=estimate(snapshot, options)["suggestions"])
    started = time.time()
    hashes = {}
    temporaries = []
    final_png = None
    try:
        canvas, placed = render(snapshot, options, layout, cancelled=cancelled, progress=progress)
        for item in placed:
            hashes[item["cell"]] = item.get('source_sha256') or _source_sha256(next(cell["path"] for cell in snapshot["cells"] if cell["id"] == item["cell"]))
        manifest = build_manifest(snapshot, options, layout, placed, export_id, hashes)
        manifest["created_at"] = datetime.now().isoformat(timespec="seconds")
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8")
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        if cancelled():
            raise ExportCancelled()
        info, overflow = png_metadata(snapshot, layout, options, export_id, manifest_sha, placed)
        png_temporary = Path(directory) / f".{stem}-{os.urandom(8).hex()}.tmp"
        manifest_temporary = Path(directory) / f".{stem}-{os.urandom(8).hex()}.json.tmp"
        temporaries += [png_temporary, manifest_temporary]
        manifest_temporary.write_bytes(manifest_bytes)
        canvas.save(png_temporary, format="PNG", pnginfo=info)
        del canvas
        if cancelled():
            raise ExportCancelled()
        final_png = Path(publish(png_temporary))
        try:
            _publish_sibling(manifest_temporary, final_png.with_suffix(".json"))
        except BaseException:
            final_png.unlink(missing_ok=True)
            final_png = None
            raise
        return {"png": final_png, "manifest": final_png.with_suffix(".json"), "width": layout["width"], "height": layout["height"],
                "percent": layout["percent"], "bytes": final_png.stat().st_size, "manifest_sha256": manifest_sha,
                "metadata_overflow": overflow, "warnings": layout["warnings"], "elapsed_seconds": round(time.time() - started, 2),
                "placed": len(placed)}
    finally:
        for path in temporaries:
            Path(path).unlink(missing_ok=True)
