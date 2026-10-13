// Entry registry: mode changes are UI-only, never apply/unload/cancel/submit.
export const GENERATION_MODES = Object.freeze([
  Object.freeze({ id: "basic", label: "基础生图", capabilitiesReady: true, executorReady: true, component: "basic", restore: "engineSettingsByEngine" }),
  // Phase B persistent experiments + Phase C compatible-cell batches + Phase D whole-grid PNG export +
  // Phase E Viewer/Gallery/apply-to-Basic integration. Independent review/QA is still outstanding.
  Object.freeze({ id: "xyz_plot", label: "XYZ 对比生成", capabilitiesReady: true, executorReady: true, component: () => import("./XYZPlotWorkspace.jsx"), restore: "xyzByEngine", phase: "E · 集成" }),
  // QPlan2/P2: bounded LoRA-weight search. It compiles to the XYZ executor but owns its own draft, rules and sessions.
  Object.freeze({ id: "lora_mix_search", label: "LoRA 混合搜索", capabilitiesReady: true, executorReady: true, component: () => import("./LoraMixWorkspace.jsx"), restore: "loraMixByEngine", phase: "P2 · 会话 / 候选网格" }),
]);
export function generationMode(id) { return GENERATION_MODES.find((mode) => mode.id === id); }
export function stepGenerationMode(id, direction) {
  const index = GENERATION_MODES.findIndex((mode) => mode.id === id);
  return GENERATION_MODES[(Math.max(0, index) + (direction < 0 ? -1 : 1) + GENERATION_MODES.length) % GENERATION_MODES.length].id;
}
