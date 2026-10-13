export const TAGGER_ENGINES = [
  { id: "wd", name: "WD · SmilingWolf", default: "wd-swinv2-v3", defaults: { general: .35, character: .85 } },
  { id: "cl", name: "CL Tagger v1.02", default: "cl-tagger-v1.02", defaults: { general: .9, character: .9, copyright: .9, artist: .9, meta: .9, tag_model: .9, quality: .9 } },
  { id: "camie", name: "Camie Tagger v2", default: "camie-tagger-v2", defaults: { general: .5, character: .5, copyright: .5, artist: .5, meta: .5, year: .5 } },
  { id: "pixai", name: "PixAI Tagger v1.0", default: "pixai-tagger-v1.0", defaults: { general: .17, character: .27, copyright: .24, style: .15, meta: .17 } },
];
export const TAGGER_CATEGORY_LABELS = { general: "常规", character: "角色", copyright: "作品", artist: "作者", meta: "元数据", year: "年份", style: "风格", tag_model: "模型类别", model: "模型类别", quality: "质量" };
export function taggerDefaults(engine) { return { ...TAGGER_ENGINES.find(row => row.id === engine).defaults }; }
export function taggerParameters(engine, model, device, thresholds, limit, alpha) {
  return { engine, model, device, ...Object.fromEntries(Object.keys(taggerDefaults(engine)).map(key => [key, Number(thresholds[key])])), limit: Number(limit), alpha: Boolean(alpha) };
}
export function resultSource(entry) {
  const params = entry?.result?.params || entry?.params;
  if (!params) return "等待图片";
  const runtime = entry.result?.runtime;
  return `${params.model} · ${runtime?.name || "ONNX Runtime"} / ${runtime?.provider || "CPUExecutionProvider"}`;
}
export function popoverPosition(anchor, width, height, viewportWidth, viewportHeight) {
  const gap = 8, margin = 12;
  const left = Math.max(margin, Math.min(anchor.left, viewportWidth - width - margin));
  const below = anchor.bottom + gap;
  const top = below + height <= viewportHeight - margin ? below : Math.max(margin, anchor.top - height - gap);
  return { left, top };
}
