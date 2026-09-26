/**
 * What the performance panel starts with, and what "恢复建议值" restores to.
 *
 * The table below is the shipped baseline and mirrors `DEFAULT_PERFORMANCE_SETTINGS` in
 * `backend/performance_settings.py`; the server is the one that decides what an untouched install
 * actually runs with, and this copy is what the panel shows before the first answer arrives and
 * what the restore button and the preset cards are built from.
 *
 * One entry is not a constant. The staged decode -- sample, push the model back to system memory,
 * then load the autoencoder on its own -- costs a reload between runs and is not worth it on a card
 * with room for both at once. Below the ceiling it is the difference between a large canvas
 * decoding and a large canvas failing, so that is where it ships on. It stays a default and never
 * becomes a lock: the toggle is always live, and a saved `false` survives.
 */

export const DEFAULT_PERFORMANCE = {
  memory_mode: "auto",
  attention_backend: "auto",
  compute_dtype: "fp16",
  vae_mode: "auto",
  cuda_math: "balanced",
  keep_model_cached: true,
  allow_shared_memory: true,
  calculate_model_hash: false,
  staged_vae_decode: false,
  compile_transformer: false,
  vram_limit_gb: 0,
};

/**
 * Cards at or below this size start with the staged decode on. Half a gigabyte of tolerance, the
 * same as the preset ranges use: a card sold as "16 GB" reports a little under 16 GiB, and the next
 * size up is 20 GB, so nothing larger can drift in under the ceiling.
 *
 * Mirrors `STAGED_DECODE_VRAM_CEILING_GB` in `backend/performance_settings.py`.
 */
export const STAGED_DECODE_VRAM_CEILING_GB = 16.5;

export function stagedDecodeDefault(vramGb) {
  const total = Number(vramGb);
  return Number.isFinite(total) && total > 0 && total <= STAGED_DECODE_VRAM_CEILING_GB;
}

/** The baseline for a machine with this much VRAM. Unknown VRAM keeps the plain table. */
export function performanceDefaultsFor(vramGb) {
  return { ...DEFAULT_PERFORMANCE, staged_vae_decode: stagedDecodeDefault(vramGb) };
}

/**
 * The recommended-preset cards.
 *
 * Every preset except the adaptive one states its own intent and is the same on every machine --
 * `speed` keeps the model resident precisely because it is for a card that can afford to, and the
 * rest ask for the staged decode outright. Only `auto` follows the card, which is also what keeps
 * exactly one card highlighted as active on a fresh install.
 */
export function performancePresetsFor(vramGb) {
  return [
    { id: "auto", label: "AUTO · 动态决策", range: "推荐 6 GB 以上", detail: "每个阶段按实时显存重新决定驻留路径，8 GB 手动配置不受影响", settings: performanceDefaultsFor(vramGb) },
    { id: "speed", label: "高速直通", range: "12 GB 以上", detail: "优先让模型完整驻留 GPU，适合连续生成", settings: { ...DEFAULT_PERFORMANCE, memory_mode: "high_vram", vae_mode: "full" } },
    { id: "large-model", label: "大型模型调度", range: "8 GB 以上", detail: "大型模型采样后转入内存，再单独执行 VAE 解码", settings: { ...DEFAULT_PERFORMANCE, memory_mode: "sdxl_balanced", staged_vae_decode: true } },
    { id: "balanced", label: "显存平衡", range: "4 GB 以上", detail: "按组件迁移，采样后释放显存并切片解码", settings: { ...DEFAULT_PERFORMANCE, memory_mode: "normal_vram", vae_mode: "sliced", staged_vae_decode: true } },
    { id: "low-memory", label: "低占用保护", range: "不足 4 GB", detail: "逐层卸载、分块解码，并在任务结束后释放模型", settings: { ...DEFAULT_PERFORMANCE, memory_mode: "low_vram", attention_backend: "sliced", vae_mode: "tiled", keep_model_cached: false, staged_vae_decode: true } },
    { id: "ultra-low", label: "极限省存 · Illustrious", range: "4 GB 显存 + 16 GB 内存", detail: "串行计算正负条件并关闭附加链路，目标是让 1024 画布尽量跑通", settings: { ...DEFAULT_PERFORMANCE, memory_mode: "ultra_low_vram", attention_backend: "sliced", compute_dtype: "fp16", vae_mode: "tiled", cuda_math: "strict", keep_model_cached: false, allow_shared_memory: false, staged_vae_decode: true } },
    { id: "large-canvas", label: "高分辨率稳态", range: "大画布", detail: "预留更多显存并使用空间分块，减少峰值波动", settings: { ...DEFAULT_PERFORMANCE, vae_mode: "tiled", allow_shared_memory: false, staged_vae_decode: true } },
  ];
}
