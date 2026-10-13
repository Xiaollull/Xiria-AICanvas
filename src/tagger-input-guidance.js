// Entry-owned manual guidance only. Never schedules a tag, install or retry.
export function inputAvailability(selection, catalog, busy = false) {
  const row = catalog?.models.find(model => model.id === selection.model);
  const prefix = `${selection.model} · `;
  if (!catalog || row?.verification_pending || (selection.device === "gpu" && row?.devices?.gpu?.ort_layer?.pending)) {
    return { kind: "checking", text: `${prefix}正在等待完整性校验；请等待校验完成后点击开始反推，图片已保留，无需因校验中重新安装。` };
  }
  if (row?.ready === false) return { kind: "blocked", text: `${prefix}${row.readiness_reason || "该引擎当前不可运行"}；图片已保留，请先解决运行条件。` };
  if (!row?.installed) {
    const reason = row?.installation_reason || "";
    if (reason && !reason.includes("缺失")) return { kind: "invalid", text: `${prefix}完整性校验未通过：${reason}。请核查文件后重新校验，图片已保留。` };
    return { kind: "missing", text: `${prefix}${reason ? "模型文件缺失或目录不可用；请先核查目录，缺少文件时再安装所选模型" : "所选模型未安装，请先安装模型"}，再点击开始反推。` };
  }
  if (selection.device === "gpu" && row?.devices?.gpu?.available !== true) return { kind: "blocked", text: `${prefix}${row?.devices?.gpu?.reason || "GPU 运行条件当前不可用"}；图片已保留，不会自动改用 CPU。` };
  if (busy) return { kind: "busy", text: `${prefix}模型安装或文件夹任务正在运行；图片已保留，请稍后点击开始反推。` };
  return { kind: "ready", text: `${prefix}校验完成，可点击开始反推；图片已保留，不会自动重跑。` };
}

export function initialInputGuidance(selection, catalog, busy) {
  const value = inputAvailability(selection, catalog, busy);
  return { notice: value.text, inputGuidance: { selection: { ...selection }, notice: value.text, kind: value.kind } };
}

export function updateInputGuidance(entry, selection, catalog, busy) {
  const owned = entry?.inputGuidance;
  if (!owned || entry.status !== "idle" || entry.result || entry.error || entry.notice !== owned.notice) return null;
  if (["engine", "model", "device"].some(key => owned.selection[key] !== selection[key])) return null;
  const value = inputAvailability(selection, catalog, busy);
  if (value.text === owned.notice) return null;
  return { notice: value.text, inputGuidance: { ...owned, notice: value.text, kind: value.kind } };
}
