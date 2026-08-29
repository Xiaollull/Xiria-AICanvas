// Pure presentation helpers for the settings plugin page.
//
// The registry answers with stable machine codes only, so every user-facing string is produced here
// and stays testable without a browser.

export const PLUGIN_STATE_PRESENTATION = {
  discovered: { label: "可用", tone: "ready", hint: "清单校验通过，可以启用。" },
  blocked: { label: "已阻止", tone: "blocked", hint: "清单声明了权限；本版本不支持任何权限，无法启用。" },
  incompatible: { label: "不兼容", tone: "warning", hint: "清单声明的宿主 API 区间不包含当前版本。" },
  invalid: { label: "无效", tone: "error", hint: "清单缺失或未通过校验。" },
};

const DIAGNOSTIC_MESSAGES = {
  plugins_root_unavailable: "无法读取 plugins 目录，请检查文件权限。",
  plugins_root_unsafe: "plugins 不是普通目录，或它是链接 / junction / 重解析点。",
  invalid_plugin_id: "文件夹名不是合法插件 ID：需要小写 kebab-case，长度 3–64。",
  duplicate_id: "存在仅大小写不同的重复文件夹，双方都已失效。",
  manifest_missing: "缺少 plugin.json。",
  manifest_unreadable: "无法读取 plugin.json。",
  manifest_too_large: "plugin.json 超过 64 KiB。",
  manifest_not_utf8: "plugin.json 不是 UTF-8，或包含 BOM，请另存为「UTF-8 无 BOM」。",
  manifest_not_json: "plugin.json 不是合法 JSON 对象（注意不能写注释和尾逗号）。",
  manifest_changed_during_read: "读取过程中 plugin.json 被替换或删除。",
  unsafe_reparse_point: "插件目录或 plugin.json 是链接 / junction / 重解析点。",
  invalid_manifest: "plugin.json 含未知字段、类型错误或超出取值范围。",
  id_folder_mismatch: "plugin.json 的 id 与所在文件夹名不一致。",
  unsupported_schema_version: "schemaVersion 不受支持，当前只接受 1。",
  invalid_version: "version 不是严格的 major.minor.patch。",
  invalid_entrypoint: "entrypoints 路径违规，必须是插件目录内的相对路径。",
  host_api_incompatible: "hostApi 区间不包含当前宿主 API 版本。",
  permissions_not_supported: "permissions 非空；本版本不支持任何权限声明。",
  plugin_state_unreadable: "state-cache/plugins.json 无法解析，已按全部禁用处理，且不会被覆盖。",
  plugin_unavailable: "插件不可用，请检查文件权限与磁盘状态。",
};

export function pluginStatePresentation(state) {
  return PLUGIN_STATE_PRESENTATION[state] || { label: "未知", tone: "error", hint: "" };
}

export function pluginDiagnosticMessage(code) {
  return DIAGNOSTIC_MESSAGES[code] || `未识别的诊断码：${code}`;
}

/**
 * A plugin can only be switched on when its manifest is valid and compatible. Because the registry
 * reports `enabled` as `false` for anything that is not `discovered`, the same condition also
 * governs switching off, so the control is interactive exactly when the state is `discovered`.
 */
export function pluginToggleAvailable(entry) {
  return Boolean(entry) && entry.state === "discovered";
}

/**
 * Text for the irreversible removal confirmation. Removing deletes the folder and everything in it,
 * so the prompt names the plugin, its folder, and the consequence explicitly.
 */
export function pluginRemoveConfirmation(entry) {
  return `确定移除插件「${entry.name}」吗？\n\n这会删除 plugins/${entry.id} 文件夹及其中的全部文件，操作不可恢复。`;
}

export function pluginRegistrySummary(registry) {
  const plugins = registry?.plugins || [];
  return {
    total: plugins.length,
    enabled: plugins.filter((entry) => entry.enabled).length,
    needsAttention: plugins.filter((entry) => entry.state !== "discovered").length,
  };
}
