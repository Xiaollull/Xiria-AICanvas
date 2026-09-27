import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { quantizationKitchenSummary, quantizationStatusRows } from "../src/quantization-status.js";

// The shape `backend/comfy_quant.py::quantization_capabilities` publishes on an RTX 4060 (SM 8.9)
// without comfy-kitchen installed.
const capabilities = {
  quantization: {
    reference: "ComfyUI 0.33.0 / comfy-kitchen 0.2.31",
    comfy_kitchen: { available: false, version: null, cuda: false, reason: "not installed" },
    formats: {
      NVFP4: { format: "nvfp4", path: "emulated", hardware_native: false, execution: "emulated: ..." },
      "ConvRot W4A4 INT4": { format: "convrot_w4a4", path: "integer", kernel: "ported kernel", weight_bits: 4, activation_bits: 4, hardware_native: true },
      "FP8 E4M3": { format: "float8_e4m3fn", path: "dequantized", hardware_native: true },
      "ConvRot INT8": { format: "int8_tensorwise", path: "integer", kernel: "ported kernel", weight_bits: 8, activation_bits: 8, hardware_native: true },
      "Future Layout": { format: "x", path: "unknown", execution: "whatever the backend says" },
    },
  },
};

test("each layout is described by its execution path, familiar formats first", () => {
  const rows = quantizationStatusRows(capabilities);
  assert.deepEqual(rows.map((row) => row.label), ["FP8 E4M3", "ConvRot INT8", "ConvRot W4A4 INT4", "NVFP4", "Future Layout"]);
  assert.equal(rows[1].detail, "W8A8 整数矩阵乘 · PyTorch 移植实现");
  assert.equal(rows[2].detail, "W4A4 整数矩阵乘 · PyTorch 移植实现");
  const triton = { quantization: { formats: { INT8: { path: "integer", kernel: "built-in Triton", weight_bits: 8, activation_bits: 8 } } } };
  assert.equal(quantizationStatusRows(triton)[0].detail, "W8A8 整数矩阵乘 · 内置 Triton 内核");
  assert.match(rows[3].detail, /SM 10\.0\+/);
  assert.equal(rows[0].detail, "逐层解码为计算精度");
  // NVFP4 / MXFP8 say which code decodes them; a layout without a decoder field says nothing more.
  const decoded = (decoder) => quantizationStatusRows({ quantization: { formats: {
    NVFP4: { path: "emulated", decoder }, MXFP8: { path: "dequantized", decoder },
  } } });
  assert.deepEqual(decoded("built-in Triton").map((row) => row.detail), [
    "逐层解码为计算精度 · 内置 Triton 解码",
    "逐层解码为计算精度 · 内置 Triton 解码（原生内核需 SM 10.0+，与 ComfyUI 的回退一致）",
  ]);
  assert.match(decoded("ported kernel")[1].detail, /PyTorch 移植实现解码/);
  // A path the frontend does not know yet falls back to the backend's own words.
  assert.equal(rows[4].detail, "whatever the backend says");
});

test("the kitchen summary never claims an install that did not happen", () => {
  assert.match(quantizationKitchenSummary(capabilities), /PyTorch 移植实现/);
  const withTriton = (state) => ({ quantization: { ...capabilities.quantization, triton_kernels: state } });
  assert.match(quantizationKitchenSummary(withTriton({ available: true, reason: "" })), /内置 Triton 内核，结果与 PyTorch 移植实现逐位一致/);
  assert.match(quantizationKitchenSummary(withTriton({ available: null, reason: "verified on first use" })), /首次使用时先/);
  assert.match(quantizationKitchenSummary(withTriton({ available: false, reason: "no CUDA device" })), /未启用：no CUDA device/);
  const installed = { quantization: { ...capabilities.quantization, comfy_kitchen: { available: true, cuda: true, version: "0.2.31" } } };
  assert.match(quantizationKitchenSummary(installed), /comfy-kitchen 0\.2\.31/);
  assert.equal(quantizationKitchenSummary({}), "");
  assert.deepEqual(quantizationStatusRows({}), []);
  assert.deepEqual(quantizationStatusRows(null), []);
});

test("the performance panel renders the rows", async () => {
  const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.match(app, /quantizationStatusRows\(performanceCapabilities\)/);
  assert.match(app, /className="nvfp4-status quant-status"/);
});
