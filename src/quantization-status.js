// P9: what the performance panel says about ComfyUI's quantized layouts on this machine. The
// backend publishes `capabilities.quantization` (`backend/comfy_quant.py::quantization_capabilities`);
// this turns its stable `path` codes into the panel's rows, so the wording lives in one place.

const KERNEL_LABELS = {
  "comfy-kitchen CUDA": "comfy-kitchen CUDA 内核",
  "built-in Triton": "内置 Triton 内核",
  "ported kernel": "PyTorch 移植实现",
};

// NVFP4 / MXFP8 name the code that decodes them; the scalar-scale formats have no decoder field.
const DECODER_LABELS = {
  "built-in Triton": "内置 Triton 解码",
  "ported kernel": "PyTorch 移植实现解码",
};
const decoderNote = (info) => (info.decoder ? ` · ${DECODER_LABELS[info.decoder] || info.decoder}` : "");

const PATH_LABELS = {
  integer: (info) => `W${info.weight_bits}A${info.activation_bits} 整数矩阵乘 · ${KERNEL_LABELS[info.kernel] || info.kernel || "PyTorch 移植实现"}`,
  emulated: (info) => `逐层解码为计算精度${decoderNote(info)}（原生内核需 SM 10.0+，与 ComfyUI 的回退一致）`,
  dequantized: (info) => `逐层解码为计算精度${decoderNote(info)}`,
};

// The order a reader expects: the familiar formats first, the 4-bit ones last.
const LAYOUT_ORDER = ["FP8 E4M3", "FP8 E5M2", "INT8", "ConvRot INT8", "ConvRot W4A4 INT4", "MXFP8", "NVFP4"];

export function quantizationStatusRows(capabilities) {
  const formats = capabilities?.quantization?.formats;
  if (!formats || typeof formats !== "object") return [];
  const labels = Object.keys(formats).sort((first, second) => {
    const a = LAYOUT_ORDER.indexOf(first);
    const b = LAYOUT_ORDER.indexOf(second);
    return (a < 0 ? LAYOUT_ORDER.length : a) - (b < 0 ? LAYOUT_ORDER.length : b) || first.localeCompare(second, "en");
  });
  return labels.map((label) => {
    const info = formats[label] || {};
    const describe = PATH_LABELS[info.path];
    return { label, detail: describe ? describe(info) : String(info.execution || ""), native: Boolean(info.hardware_native) };
  });
}

export function quantizationKitchenSummary(capabilities) {
  const quantization = capabilities?.quantization;
  const kitchen = quantization?.comfy_kitchen;
  if (!kitchen) return "";
  if (kitchen.available && kitchen.cuda) return `已检测到 comfy-kitchen ${kitchen.version}，INT8 / W4A4 使用其 CUDA 内核。`;
  const triton = quantization.triton_kernels;
  if (triton && triton.available !== false) {
    return triton.available
      ? "INT8 / W4A4 使用内置 Triton 内核，结果与 PyTorch 移植实现逐位一致。"
      : "INT8 / W4A4 将使用内置 Triton 内核；首次使用时先与 PyTorch 移植实现逐位比对，通过后启用。";
  }
  return `INT8 / W4A4 使用 PyTorch 移植实现（结果与 comfy-kitchen eager 后端一致）${triton?.reason ? `；内置 Triton 内核未启用：${triton.reason}` : ""}。`;
}
