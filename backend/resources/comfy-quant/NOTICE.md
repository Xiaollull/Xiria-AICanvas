# ComfyUI Quantized-Layout Port (P9)

`backend/comfy_quant.py` ports how ComfyUI reads and executes its quantized checkpoint layouts, so a
file means the same thing here as in ComfyUI. Parts of `backend/quantized_linear.py` and
`backend/flux_pipeline.py` call into it. Nothing below is imported at runtime from ComfyUI or
comfy-kitchen; when the pinned comfy-kitchen happens to be installed with a usable CUDA backend,
its two integer matmuls are called instead of the ports.

## ComfyUI

- Version: 0.33.0 (the user's install at `F:\AI\Ai ComfyUi\ComfyUI`)
- Copyright: Comfy Org and ComfyUI contributors
- License: GNU General Public License v3.0. XiriaCanvas AI is distributed under the GNU Affero
  General Public License v3.0; the combined work's source is published with every release.
- Ported: `comfy/quant_ops.py` (`QUANT_ALGOS` format table), `comfy/ops.py`
  (`_load_quantized_module` marker and scale reading, `mixed_precision_ops` choice between the
  quantized and full-precision matmul), `comfy/utils.py` (`convert_old_quants`, the
  `_quantization_metadata` header table), `comfy/model_management.py`
  (`supports_fp8_compute`, `supports_nvfp4_compute`, `supports_mxfp8_compute`).

## comfy-kitchen

- Version: 0.2.31 (the version ComfyUI 0.33.0 pins in `requirements.txt`)
- Source: PyPI `comfy_kitchen-0.2.31-py3-none-any.whl`, SHA-256
  `5117946c30f308cfc73b9c26f723ae3918308bd090e57a8eae298406934aabd6`
- Copyright: Comfy Org. Its NVFP4 decoding derives from PyTorch AO (BSD-3-Clause, Meta Platforms).
- License: Apache License 2.0, included here as `LICENSE-APACHE-2.0.txt`; its NOTICE file is
  included unchanged as `NOTICE-comfy-kitchen.txt`.
- Ported: `tensor/int8.py`, `tensor/int8_utils.py`, `tensor/convrot_w4a4.py`,
  `backends/eager/quantization.py` (INT8 row-wise quantization, integer matmul padding, INT8 linear,
  ConvRot rotation, NVFP4 and MXFP8 decoding), `backends/eager/convrot_w4a4.py`,
  `backends/eager/svdquant.py` (INT4 packing), `float_utils.py` (cuBLAS block-scale tile layout,
  E8M0 decoding).

## Built-in Triton kernels

`backend/comfy_quant_triton.py` follows the structure of comfy-kitchen 0.2.31's Triton backend
(`backends/triton/quantization.py`: a one-program-per-row quantizer and a grouped-M int8 matmul with
a fused dequantizing epilogue; comfy-kitchen credits dxqb/OneTrainer for it). The code is written for
this project: its arithmetic reproduces the PyTorch port rather than comfy-kitchen's Triton kernels,
it reads the int8 weight in its stored layout, and it unpacks ConvRot W4A4's packed int4 weights in
registers instead of into memory.

## Modifications

- The kernels are plain functions rather than `torch.library` custom ops and registry entries.
- Input-activation fusion (`input_act`) and stochastic rounding are left out; nothing here uses them.
- ConvRot W4A4 accumulates its integer matmul exactly in int32 and chooses int4 or int8 activations
  the way comfy-kitchen's CUDA backend does (int4 on SM 8.x), where the eager reference multiplies
  unpacked integers in the output dtype.
- INT4 unpacking uses sign-extending int8 shifts; the decoded values are identical.
- Markers are validated strictly: unknown fields, a rotation on a layout that cannot carry one, and
  exporter fields that disagree with the stored tensors are refused instead of ignored.
