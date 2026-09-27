# Model Directory

This directory follows ComfyUI's model-category naming while remaining fully
owned by this project. The inference backend resolves every configured path
relative to the project root on both Windows and Linux.

Every root in `model-paths.json` may use custom folder names and arbitrary
nested project-relative paths, but must remain below `models`. Checkpoints,
LoRAs, upscalers, YOLO `.pt` detectors, and compatible custom background-removal
ONNX files are discovered recursively. LoRA root files appear under `根目录`;
each custom first-level folder becomes a UI category and deeper folders remain
visible in the model name. Moving or renaming a selected model changes its
root-relative ID, so refresh and reselect it after reorganizing files.

## Current structure

```text
models/
|-- checkpoints/
|   |-- sd/
|   `-- illustrious/
|-- loras/
|   |-- sd/
|   |   |-- character/
|   |   |-- style/
|   |   |-- concept/
|   |   `-- other/
|   `-- illustrious/
|       |-- character/
|       |-- style/
|       |-- concept/
|       `-- other/
|-- vae/
|-- diffusion_models/
|-- text_encoders/
|-- embeddings/
|-- yolo/
|-- yolo-models.json
|-- background-removal/
|-- background-removal-models.json
|-- upscalers/
`-- configs/
```

- `checkpoints/sd`: SD 1.x or SDXL single-file checkpoints. SDXL detection is
  supported for `.safetensors`; ambiguous SD 2.x and SDXL `.ckpt` files are not
  currently supported.
- `checkpoints/illustrious`: Illustrious/Illustrious XL checkpoints.
- `loras/sd`: SD LoRA root. The four default categories remain available, while
  custom folders and arbitrary nesting are discovered automatically.
- `loras/illustrious`: Illustrious LoRA root with the same recursive behavior.
- `vae`: Standalone and shared VAEs used by split model pipelines;
  `.safetensors`, plus `.gguf` for FLUX.1, FLUX.2 and Krea 2.
- `diffusion_models`: Anima, Flux, FLUX.2, Krea 2, and other split diffusion
  weights. `.safetensors` throughout, plus `.gguf` for the three engines that
  read it.
- `text_encoders`: Standalone text encoders required by split model pipelines;
  `.safetensors`, plus llama.cpp `.gguf` T5 / Qwen3 / Qwen3-VL encoders for
  FLUX.1, FLUX.2 and Krea 2.
- `embeddings`: Reserved for future textual inversion support.
- `yolo`: Project-owned Ultralytics `.pt` detection or segmentation models for
  local ADetailer passes. Community models may be placed here, including in
  subdirectories, and are discovered automatically.
- `yolo-models.json`: Pinned catalog for the official ADetailer starter models.
- `background-removal`: Project-owned compatible ONNX foreground extraction models.
  Recommended files are identified by fixed hashes even after nesting/renaming;
  compatible custom ONNX files are discovered recursively.
- `background-removal-models.json`: Pinned catalog for recommended transparent
  background models. Model weights are never included in program updates.
- `upscalers`: Hires.fix pixel-space super-resolution weights. Compatible
  `.pth`, `.pt`, `.ckpt`, and `.safetensors` files are discovered recursively
  by their tensor structure, so renaming a file does not change compatibility.
  PyTorch files must contain weights/state dictionaries; TorchScript archives
  are intentionally not loaded. Recommended Real-ESRGAN weights are available
  from the model downloader.
- `configs`: User/model runtime configuration artifacts. Native Anima tokenizer
  files are program resources under `backend/resources/anima-tokenizers`, not
  user model downloads.

ADetailer detection runs on CPU in this project's verified Python environment.
The UI benchmarks Hugging Face and hf-mirror routes before downloading an
official model, and resumes interrupted downloads in `models/yolo`. Detail
inpainting stays in the same project environment and inherits the active
checkpoint, LoRAs, sampler, and scheduler. Model files are never uploaded.

The current inference backend supports single-file `.safetensors` checkpoints
and SD 1.x `.ckpt` checkpoints, with `.safetensors` preferred. Illustrious
models must be placed under `checkpoints/illustrious` so the correct SDXL
pipeline is selected.

Anima, FLUX.1, FLUX.2, and Krea 2 use `diffusion_models`, `text_encoders`, and
`vae`. All four generate natively and each needs one compatible file from every
directory it mounts. The Anima Qwen3 and T5 tokenizer JSON files ship with the
program and are integrity-checked during setup, startup and updates.

FLUX.1, FLUX.2, and Krea 2 also load every component from `.gguf`. A llama.cpp
text encoder is renamed back into ComfyUI's tensor names (`t5encoder` for
FLUX.1, `qwen3` for FLUX.2, `qwen3vl` for Krea 2) and refused if its family is
not the engine's; a Mistral (`llama`) GGUF is refused by name. A GGUF is
expanded to the compute dtype while loading, so it is a smaller download rather
than a smaller model in memory.

ComfyUI's quantized safetensors layouts — FP8, INT8 and ConvRot INT8, ConvRot
W4A4 INT4, MXFP8 and NVFP4, marked per layer or in the header's
`_quantization_metadata` — are read the way ComfyUI 0.33.0 reads them. Krea 2
keeps them in their stored form (INT8/W4A4 run ComfyUI's integer matmul, on built-in Triton kernels
where available;
NVFP4/MXFP8 are dequantized per layer below SM 10.0, as ComfyUI does);
FLUX.1/FLUX.2 expand them at load. The ComfyUI-INT8-Fast custom node's
format-less marker and `asym_w4a8_int8` are refused by name. See
`docs/models.html#quantization` for the full matrix.
