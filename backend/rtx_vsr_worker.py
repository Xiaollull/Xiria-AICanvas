import argparse
import json
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import torch
from PIL import Image


QUALITY_NAMES = {
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "ultra": "ULTRA",
}
WORKER_PROTOCOL = 1
MAX_OUTPUT_EDGE = 8192
MAX_OUTPUT_PIXELS = 32 * 1024 * 1024


def video_super_resolution(tensor, width, height, quality):
    import nvvfx

    quality_level = getattr(nvvfx.effects.QualityLevel, QUALITY_NAMES[quality])
    with nvvfx.VideoSuperRes(quality_level) as effect:
        effect.output_width = width
        effect.output_height = height
        effect.load()
        return torch.from_dlpack(effect.run(tensor).image).clone()


def validate_output(output, width, height):
    if output.shape != (3, height, width):
        raise RuntimeError(f"RTX VSR returned shape {tuple(output.shape)}, expected {(3, height, width)}")
    if output.device.type != "cuda" or output.dtype != torch.float32:
        raise RuntimeError("RTX VSR returned an invalid tensor type or device")
    if not torch.isfinite(output).all():
        raise RuntimeError("RTX VSR returned non-finite pixels")


def run_probe():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    tensor = torch.linspace(0, 1, 3 * 64 * 64, device="cuda", dtype=torch.float32).reshape(3, 64, 64)
    output = video_super_resolution(tensor, 128, 128, "low")
    validate_output(output, 128, 128)
    return {
        "protocol": WORKER_PROTOCOL,
        "available": True,
        "runtime_version": package_version("nvidia-vfx"),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "compute_capability": ".".join(str(value) for value in torch.cuda.get_device_capability()),
        "torch": torch.__version__,
    }


def run_image(input_path, output_path, width, height, quality):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if min(width, height) < 8 or max(width, height) > MAX_OUTPUT_EDGE or width * height > MAX_OUTPUT_PIXELS:
        raise ValueError("RTX VSR output exceeds the safe size limit")
    with Image.open(input_path) as opened:
        source = opened.convert("RGB")
        pixels = np.asarray(source, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.moveaxis(pixels, 2, 0).copy()).to(device="cuda", dtype=torch.float32).contiguous()
    output = video_super_resolution(tensor, width, height, quality)
    validate_output(output, width, height)
    pixels = np.moveaxis(output.detach().float().cpu().clamp_(0, 1).numpy(), 0, 2)
    Image.fromarray(np.rint(pixels * 255).astype(np.uint8), "RGB").save(output_path)
    return {"protocol": WORKER_PROTOCOL, "available": True, "width": width, "height": height, "quality": quality}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--quality", choices=sorted(QUALITY_NAMES), default="high")
    args = parser.parse_args()
    if args.probe:
        result = run_probe()
    else:
        if not args.input or not args.output or not args.width or not args.height:
            raise ValueError("RTX VSR input, output, width, and height are required")
        result = run_image(Path(args.input), Path(args.output), args.width, args.height, args.quality)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
