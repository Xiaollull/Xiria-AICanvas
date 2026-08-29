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
- `vae`: Standalone and shared VAEs used by split model pipelines.
- `diffusion_models`: Anima, Flux, Krea 2, and other split diffusion weights.
- `text_encoders`: Standalone text encoders required by split model pipelines.
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

Anima, Flux, and Krea 2 downloads use `diffusion_models`, `text_encoders`, and
`vae`. Native Anima generation is enabled and requires one compatible file from
each directory. Its Qwen3 and T5 tokenizer JSON files ship with the program and
are integrity-checked during setup, startup and updates. Flux and Krea 2 remain
download-only until their native inference pipelines are added.
