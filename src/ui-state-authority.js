// Admission, not migration: individual schema owners retain unknown/fatal raw containers.
import { ENGINE_SETTINGS_SCHEMA_VERSION, SETTINGS_ENGINES } from "./engine-settings.js";
const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
export function assertUiStateValue(state) {
  if (!object(state)) throw new Error("项目页面状态不是可读取的对象；原数据已保留。");
  if (Object.hasOwn(state, "workspace")) {
    if (!object(state.workspace)) throw new Error("工作区状态形状无法读取；原数据已保留。");
    const map = state.workspace.engineSettingsByEngine;
    if (Object.hasOwn(state.workspace, "engineSettingsByEngine") &&
      (!object(map) || !object(map.byEngine) || !Number.isInteger(map.schemaVersion) || map.schemaVersion < 1 || map.schemaVersion > ENGINE_SETTINGS_SCHEMA_VERSION ||
        Object.entries(map.byEngine).some(([engine, record]) => !SETTINGS_ENGINES.includes(engine) || !object(record)))) {
      throw new Error("引擎参数容器版本无法安全读取；原数据已保留。");
    }
  } else if (Object.keys(state).some((key) => !["theme", "motionEnabled"].includes(key))) {
    throw new Error("项目页面状态缺少工作区；原数据已保留。");
  }
  return state;
}
export function admitUiStateResponse(payload) {
  if (!object(payload) || !Object.hasOwn(payload, "state")) throw new Error("页面状态响应不完整；未启用保存。");
  if (payload.exists === false) {
    if (payload.state !== null) throw new Error("页面状态存在性响应不一致；未启用保存。");
    return { status: "confirmed-absent", state: null };
  }
  const state = assertUiStateValue(payload.state);
  return { status: "loaded", state };
}
