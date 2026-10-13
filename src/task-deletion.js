export const TASK_SELECTION_LIMIT = 200;
export function deletionBody(scope, ids = []) {
  if (scope === "all") return { scope };
  if (scope !== "selected" || !ids.length || ids.length > TASK_SELECTION_LIMIT || new Set(ids).size !== ids.length || ids.some((id) => !/^[0-9a-f]{32}$/.test(id))) throw new Error("请选择 1–200 个合法任务编号");
  return { scope, ids: [...ids] };
}
export function deletionResult(value) {
  if (!value || !Array.isArray(value.results) || value.results.some((item) => !/^[0-9a-f]{32}$/.test(item.id) || !["deleted", "skipped", "failed"].includes(item.outcome) || !Array.isArray(item.experiment_ids))) throw new Error("服务端未返回有效删除凭据；请刷新核对，不能确认成功");
  for (const outcome of ["deleted", "skipped", "failed"]) if (value[outcome] !== value.results.filter((item) => item.outcome === outcome).length) throw new Error("删除统计与明细不一致；请刷新核对，不能确认成功");
  return value;
}
// visibilityRemoved comes only from a fresh GET after an ambiguous POST: reconcile UI absence,
// without manufacturing a successful deletion receipt or claiming who removed the task.
export const removedExperiments = (result) => result.visibilityRemoved || result.results.filter((item) => item.outcome === "deleted").flatMap((item) => item.experiment_ids);
