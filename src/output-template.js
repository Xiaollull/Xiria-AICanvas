// The output path template a workspace saves under Settings → General.
//
// The server parses, validates and resolves every template (`backend/output_paths.py`) and generates
// every file name; this module only decides what the browser stores and sends. The default template
// is the historical date-folder scheme spelled out, and it is sent as *nothing*: a request without a
// template names its files exactly as every earlier build did, `-02` collision suffixes included.
// Anything else is a relative path beneath the output root whose last segment is the file stem —
// `Test/image` writes `Test/image_0001.png`, then `_0002`.

export const DEFAULT_OUTPUT_TEMPLATE = "{date}/XirAI-{date}-{hour}-{minute}";
export const OUTPUT_TEMPLATE_MAX_LENGTH = 200;
export const OUTPUT_TEMPLATE_PLACEHOLDERS = Object.freeze([
  { name: "date", label: "日期 YYYY-MM-DD" },
  { name: "year", label: "年 YYYY" },
  { name: "month", label: "月 MM" },
  { name: "day", label: "日 DD" },
  { name: "hour", label: "时 HH（24 小时）" },
  { name: "minute", label: "分 mm" },
  { name: "second", label: "秒 ss" },
  { name: "engine", label: "引擎" },
]);

/**
 * A stored template, repaired only where it is clearly damaged.
 *
 * Returns the text to keep (the default when there is none) and a warning when a stored value had
 * to be dropped. Whether a template is *valid* is the server's call, made again at submission.
 */
export function normalizeStoredOutputTemplate(value) {
  if (value === undefined || value === null || value === "") return { value: DEFAULT_OUTPUT_TEMPLATE, warning: "" };
  if (typeof value !== "string" || value.length > OUTPUT_TEMPLATE_MAX_LENGTH * 2) {
    return { value: DEFAULT_OUTPUT_TEMPLATE, warning: "输出路径模板设置无效，已恢复默认命名" };
  }
  return { value: value.normalize("NFC"), warning: "" };
}

/** What a job carries: `null` for the historical scheme, otherwise the trimmed template. */
export function outputTemplatePayload(value) {
  const text = typeof value === "string" ? value.normalize("NFC").trim() : "";
  return !text || text === DEFAULT_OUTPUT_TEMPLATE ? null : text;
}
