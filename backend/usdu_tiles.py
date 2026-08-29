"""Pure Pillow geometry and compositing helpers for USDU-style tiled work.

Decoded tiles are already-quantized ``PIL.Image`` RGB images.  Upstream tensor
decoders own float-to-uint8 quantization; this module deliberately has no float
image entry point.
"""

from dataclasses import dataclass
import math
from typing import Literal

from PIL import Image, ImageDraw, ImageFilter


Size = tuple[int, int]
Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class TileRegion:
    """One row-major tile, using PIL's inclusive mask and exclusive crop rules."""

    mask_rect: Rect
    mask_semantic: Literal["inclusive"]
    bbox: Rect
    crop: Rect
    model_size: Size
    paste_origin: tuple[int, int]


@dataclass(frozen=True)
class TilePlan:
    canvas: Size
    core_tile: Size
    rows: int
    cols: int
    processing_size: Size
    regions: tuple[TileRegion, ...]


def _validate_size(name: str, value: Size):
    if len(value) != 2 or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise ValueError(f"{name} must contain two positive integers")


def _expand_crop(crop: Rect, canvas: Size, aspect: float) -> Rect:
    """Expand (never shrink) an exclusive crop toward the model's aspect ratio."""
    left, top, right, bottom = crop
    width, height = right - left, bottom - top
    current = width / height
    if math.isclose(current, aspect):
        return crop
    if current < aspect:
        wanted_width = min(canvas[0], max(width, round(height * aspect)))
        extra = wanted_width - width
        right = min(canvas[0], right + extra // 2)
        left = max(0, left - (extra - extra // 2))
        right = min(canvas[0], right + (wanted_width - (right - left)))
        left = max(0, left - (wanted_width - (right - left)))
    else:
        wanted_height = min(canvas[1], max(height, round(width / aspect)))
        extra = wanted_height - height
        bottom = min(canvas[1], bottom + extra // 2)
        top = max(0, top - (extra - extra // 2))
        bottom = min(canvas[1], bottom + (wanted_height - (bottom - top)))
        top = max(0, top - (wanted_height - (bottom - top)))
    return left, top, right, bottom


def plan_tiles(canvas: Size, core_tile: Size, padding=32, seam_mode="none") -> TilePlan:
    """Plan source-derived core tiles and their clamped, aspect-expanded crops."""
    _validate_size("canvas", canvas)
    _validate_size("core_tile", core_tile)
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError("padding must be a non-negative integer")
    if seam_mode != "none":
        raise ValueError("only seam_mode='none' is supported")

    canvas_width, canvas_height = canvas
    core_width, core_height = core_tile
    cols = math.ceil(canvas_width / core_width)
    rows = math.ceil(canvas_height / core_height)
    processing_size = (
        max(8, round((core_width + padding) / 8) * 8),
        max(8, round((core_height + padding) / 8) * 8),
    )
    aspect = processing_size[0] / processing_size[1]
    regions = []
    for row in range(rows):
        for col in range(cols):
            left = col * core_width
            top = row * core_height
            right = min(left + core_width, canvas_width)
            bottom = min(top + core_height, canvas_height)
            # ImageDraw.rectangle is inclusive.  The plugin supplies the core
            # end coordinate (not the exclusive crop endpoint), so neighboring
            # core tiles share their internal row and column boundary pixels.
            mask_right = min(left + core_width, canvas_width - 1)
            mask_bottom = min(top + core_height, canvas_height - 1)
            mask_rect = (left, top, mask_right, mask_bottom)
            bbox = (left, top, right, bottom)
            padded = (
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                min(canvas_width, bbox[2] + padding),
                min(canvas_height, bbox[3] + padding),
            )
            crop = _expand_crop(padded, canvas, aspect)
            regions.append(TileRegion(mask_rect, "inclusive", bbox, crop, processing_size, crop[:2]))
    return TilePlan(canvas, core_tile, rows, cols, processing_size, tuple(regions))


def prepare_tile(source: Image.Image, region: TileRegion) -> Image.Image:
    """Return one RGB model input at the plan's fixed processing size."""
    _require_canvas(source, region)
    crop = source.convert("RGB").crop(region.crop)
    if crop.size != region.model_size:
        crop = crop.resize(region.model_size, Image.Resampling.LANCZOS)
    return crop


def restore_tile(decoded: Image.Image, region: TileRegion) -> Image.Image:
    """Resize an already-quantized RGB decoded tile to its source crop."""
    if not isinstance(decoded, Image.Image):
        raise TypeError("decoded tile must be a PIL image")
    if decoded.mode != "RGB":
        raise ValueError("decoded tile must be an already-quantized RGB PIL image")
    result = decoded
    crop_size = (region.crop[2] - region.crop[0], region.crop[3] - region.crop[1])
    if result.size != crop_size:
        result = result.resize(crop_size, Image.Resampling.LANCZOS)
    return result


def _require_canvas(image: Image.Image, region: TileRegion):
    if not isinstance(image, Image.Image):
        raise TypeError("source must be a PIL image")
    if region.crop[2] > image.width or region.crop[3] > image.height:
        raise ValueError("region crop is outside source image")


class TileCompositor:
    """Sequential src-over compositor retaining one full-canvas RGBA image."""

    def __init__(self, source: Image.Image, plan: TilePlan, mask_blur=0):
        if not isinstance(plan, TilePlan):
            raise TypeError("plan must be a TilePlan")
        if not isinstance(source, Image.Image):
            raise TypeError("source must be a PIL image")
        if source.size != plan.canvas:
            raise ValueError("source size must match plan canvas")
        if not isinstance(mask_blur, (int, float)) or isinstance(mask_blur, bool) or not math.isfinite(mask_blur) or mask_blur < 0:
            raise ValueError("mask_blur must be a finite non-negative number")
        self.plan = plan
        self.mask_blur = mask_blur
        self._canvas = source.convert("RGBA")
        self._next_index = 0
        self._used_regions = set()

    def composite(self, decoded: Image.Image, region: TileRegion | None = None):
        if region is None:
            if self._next_index >= len(self.plan.regions):
                raise ValueError("all planned tiles have already been composited")
            region = self.plan.regions[self._next_index]
        elif region not in self.plan.regions:
            raise ValueError("region does not belong to this plan")
        if region in self._used_regions:
            raise ValueError("region has already been composited")
        expected = self.plan.regions[self._next_index]
        if region != expected:
            raise ValueError("regions must be composited in row-major plan order")
        tile = restore_tile(decoded, region)
        mask = Image.new("L", self.plan.canvas, 0)
        ImageDraw.Draw(mask).rectangle(region.mask_rect, fill=255)
        if self.mask_blur:
            mask = mask.filter(ImageFilter.GaussianBlur(self.mask_blur))
        alpha = mask.crop(region.crop)
        tile.putalpha(alpha)
        self._canvas.alpha_composite(tile, dest=region.paste_origin)
        self._used_regions.add(region)
        self._next_index += 1

    def finish(self) -> Image.Image:
        return self._canvas.convert("RGB")


def composite_tiles(source: Image.Image, plan: TilePlan, decoded_tiles, mask_blur=0) -> Image.Image:
    """Composite decoded tiles in row-major order with Pillow src-over semantics."""
    try:
        tiles = tuple(decoded_tiles)
    except TypeError as error:
        raise TypeError("decoded_tiles must be an iterable of RGB PIL images") from error
    if len(tiles) != len(plan.regions):
        raise ValueError("decoded tile count must match planned regions")
    compositor = TileCompositor(source, plan, mask_blur)
    for region, decoded in zip(plan.regions, tiles):
        compositor.composite(decoded, region)
    return compositor.finish()
