// Provider profiles for the AI assistant client.
//
// Every provider listed here speaks the OpenAI `/chat/completions` wire format, which is the only
// transport the assistant proxy implements. This table exists because "inference strength" is not a
// single parameter: ordinary chat models take `temperature` over ranges that differ per provider,
// while reasoning models reject `temperature` outright and take `reasoning_effort` instead. Sending
// the wrong one is a hard 400 from the provider, so the parameter is resolved from provider *and*
// model before a request body is ever assembled.

export const ASSISTANT_SETTINGS_SCHEMA_VERSION = 1;

const TEMPERATURE = "temperature";
const REASONING_EFFORT = "reasoning_effort";

export const MAXIMUM_BASE_URL_LENGTH = 400;
export const MAXIMUM_MODEL_LENGTH = 120;
export const MAXIMUM_API_KEY_LENGTH = 400;

// Labels are the exact API tokens rather than translations: this control writes a value straight
// into the request body, so showing anything other than what is sent invites mismatched bug
// reports. Vendor names, model ids and strength parameters stay English throughout the UI.
const effortChoices = (...values) => Object.freeze(values.map((value) => Object.freeze({ value, label: value })));

// Each generation accepts a different set, and sending a tier a model does not know is a 400. The
// lists therefore stay separate rather than being unioned into one permissive set.
const O_SERIES_EFFORT = effortChoices("low", "medium", "high");
const GPT5_EFFORT = effortChoices("minimal", "low", "medium", "high");
const GPT55_EFFORT = effortChoices("none", "low", "medium", "high", "xhigh");
const GPT56_EFFORT = effortChoices("none", "low", "medium", "high", "xhigh", "max");

const unsupportedStrength = (reason) => Object.freeze({
  kind: "unsupported",
  parameter: null,
  label: null,
  defaultValue: null,
  hint: reason,
});

const reasoningStrength = (choices, hint) => Object.freeze({
  kind: "enum",
  parameter: REASONING_EFFORT,
  label: "Reasoning Effort",
  choices,
  defaultValue: "medium",
  hint,
});

const temperatureStrength = (minimum, maximum, defaultValue, hint, step = 0.1) => Object.freeze({
  kind: "numeric",
  parameter: TEMPERATURE,
  label: "Temperature",
  minimum,
  maximum,
  step,
  defaultValue,
  hint,
});

// Ordered for the picker. DeepSeek leads because it is the provider this client was specified
// against; `custom` is last because it is the escape hatch, not a recommendation.
export const AI_PROVIDERS = Object.freeze([
  Object.freeze({
    id: "deepseek",
    label: "DeepSeek",
    baseUrl: "https://api.deepseek.com/v1",
    requiresApiKey: true,
    // V4 replaced the V3 line; `deepseek-chat` / `deepseek-reasoner` were retired on 2026-07-24 and
    // now survive only as aliases onto V4-Flash (non-thinking / thinking).
    models: Object.freeze(["deepseek-v4-pro", "deepseek-v4-flash"]),
    strength: temperatureStrength(0, 2, 1, "取值 0 - 2，默认 1。V4 同时支持 temperature 与思考模式，本客户端使用 temperature。"),
    overrides: Object.freeze([]),
  }),
  Object.freeze({
    id: "openai",
    label: "OpenAI",
    baseUrl: "https://api.openai.com/v1",
    requiresApiKey: true,
    models: Object.freeze(["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5", "gpt-5.4-mini", "gpt-4o", "gpt-4o-mini"]),
    strength: temperatureStrength(0, 2, 1, "取值 0 - 2，默认 1。数值越高输出越发散。"),
    // Ordered most-specific first; the first match wins.
    overrides: Object.freeze([
      Object.freeze({
        pattern: /^gpt-5\.6/i,
        strength: reasoningStrength(GPT56_EFFORT, "gpt-5.6 系列不接受 temperature，改用推理强度；关闭档延迟最低，最大档最慢。"),
      }),
      Object.freeze({
        pattern: /^gpt-5\.5/i,
        strength: reasoningStrength(GPT55_EFFORT, "gpt-5.5 系列不接受 temperature，改用推理强度。"),
      }),
      Object.freeze({
        pattern: /^gpt-5(?:$|[-_])/i,
        strength: reasoningStrength(GPT5_EFFORT, "gpt-5 不接受 temperature，改用推理强度；最小档延迟最低。"),
      }),
      // Any other gpt-5 generation falls back to the three tiers every generation accepts, so a
      // model this build has never heard of cannot be sent a tier it will reject.
      Object.freeze({
        pattern: /^gpt-5/i,
        strength: reasoningStrength(O_SERIES_EFFORT, "该 gpt-5 系列模型不接受 temperature，已使用各代通用的推理强度档位。"),
      }),
      Object.freeze({
        pattern: /^o[1-9]/i,
        strength: reasoningStrength(O_SERIES_EFFORT, "o 系列推理模型不接受 temperature，改用推理强度。"),
      }),
    ]),
  }),
  Object.freeze({
    id: "moonshot",
    label: "Moonshot Kimi",
    baseUrl: "https://api.moonshot.cn/v1",
    requiresApiKey: true,
    // The K2 line was retired on 2026-05-25 and the moonshot-v1 series is closed to new accounts.
    models: Object.freeze(["kimi-k3", "kimi-k2.7-code", "kimi-k2.6"]),
    strength: temperatureStrength(0, 1, 0.6, "取值 0 - 1，Kimi 推荐 0.6。上限低于 OpenAI，填 2 会被拒绝。国际站点为 https://api.moonshot.ai/v1。"),
    overrides: Object.freeze([]),
  }),
  Object.freeze({
    id: "zhipu",
    label: "Zhipu GLM",
    baseUrl: "https://open.bigmodel.cn/api/paas/v4",
    requiresApiKey: true,
    models: Object.freeze(["glm-5.1", "glm-5", "glm-5-turbo", "glm-4.7"]),
    strength: temperatureStrength(0, 1, 0.95, "取值 0 - 1，默认 0.95。上限低于 OpenAI。", 0.05),
    overrides: Object.freeze([]),
  }),
  Object.freeze({
    id: "dashscope",
    label: "Alibaba Qwen",
    baseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    requiresApiKey: true,
    models: Object.freeze(["qwen3.7-max", "qwen3.7-plus", "qwen3.7-flash"]),
    strength: temperatureStrength(0, 2, 0.7, "兼容模式取值 0 - 2，默认 0.7。新版工作区地址形如 https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1。"),
    overrides: Object.freeze([]),
  }),
  Object.freeze({
    id: "siliconflow",
    label: "SiliconFlow",
    baseUrl: "https://api.siliconflow.cn/v1",
    requiresApiKey: true,
    models: Object.freeze(["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1", "Qwen/Qwen2.5-72B-Instruct"]),
    strength: temperatureStrength(0, 2, 0.7, "取值 0 - 2，默认 0.7。平台模型实时更新，建议用下方按钮拉取当前列表。"),
    overrides: Object.freeze([]),
  }),
  Object.freeze({
    id: "openrouter",
    label: "OpenRouter",
    baseUrl: "https://openrouter.ai/api/v1",
    requiresApiKey: true,
    models: Object.freeze(["anthropic/claude-sonnet-5", "openai/gpt-5.1", "deepseek/deepseek-v4-pro", "qwen/qwen3.7-plus"]),
    strength: temperatureStrength(0, 2, 1, "取值 0 - 2。转发到上游模型，实际上限以该模型为准；目录变动频繁，建议拉取当前列表。"),
    overrides: Object.freeze([]),
  }),
  Object.freeze({
    id: "ollama",
    label: "Ollama (Local)",
    baseUrl: "http://localhost:11434/v1",
    requiresApiKey: false,
    models: Object.freeze(["qwen3:8b", "qwen3-coder:30b", "gpt-oss:20b", "gemma3:27b", "llama3.3:70b"]),
    strength: temperatureStrength(0, 2, 0.8, "本地推理，默认 0.8。无需 API Key；拉取列表会返回本机已安装的模型。"),
    overrides: Object.freeze([]),
  }),
  Object.freeze({
    id: "custom",
    label: "Custom (OpenAI-compatible)",
    baseUrl: "",
    requiresApiKey: false,
    models: Object.freeze([]),
    strength: temperatureStrength(0, 2, 1, "任何兼容 /chat/completions 的服务。范围以该服务文档为准。"),
    overrides: Object.freeze([]),
  }),
]);

export const DEFAULT_PROVIDER_ID = "deepseek";

const PROVIDER_BY_ID = new Map(AI_PROVIDERS.map((provider) => [provider.id, provider]));

export function providerProfile(id) {
  return PROVIDER_BY_ID.get(typeof id === "string" ? id.trim().toLowerCase() : "") || null;
}

// Resolution order is model override first, then the provider default. A blank model deliberately
// falls back to the provider default so the settings form can render a control before the user has
// typed a model name.
export function resolveStrengthParameter(providerId, model) {
  const provider = providerProfile(providerId);
  if (!provider) return unsupportedStrength("未知的服务商，无法确定推理强度参数。");
  const name = typeof model === "string" ? model.trim() : "";
  if (name) {
    for (const override of provider.overrides) {
      if (override.pattern.test(name)) return override.strength;
    }
  }
  return provider.strength;
}

export function strengthChoiceValues(resolved) {
  return resolved?.kind === "enum" ? resolved.choices.map((choice) => choice.value) : [];
}

// Rounds to the profile's step so a slider that emits 1.3000000000000003 stores 1.3.
function quantize(value, step) {
  if (!Number.isFinite(step) || step <= 0) return value;
  const decimals = Math.max(0, String(step).includes(".") ? String(step).split(".")[1].length : 0);
  return Number((Math.round(value / step) * step).toFixed(decimals));
}

export function clampStrength(resolved, value) {
  if (!resolved || resolved.kind === "unsupported") return null;
  if (resolved.kind === "enum") {
    return strengthChoiceValues(resolved).includes(value) ? value : resolved.defaultValue;
  }
  const numeric = typeof value === "number" ? value : Number.parseFloat(String(value ?? ""));
  if (!Number.isFinite(numeric)) return resolved.defaultValue;
  return quantize(Math.min(resolved.maximum, Math.max(resolved.minimum, numeric)), resolved.step);
}

// The only place a strength value becomes request JSON. An unsupported parameter yields an empty
// object rather than a null-valued key, because providers reject the key even when it is null.
export function strengthPayload(providerId, model, strength) {
  const resolved = resolveStrengthParameter(providerId, model);
  if (resolved.kind === "unsupported") return {};
  const value = clampStrength(resolved, strength);
  if (value === null) return {};
  return { [resolved.parameter]: value };
}

// `/chat/completions` is appended rather than inferred. Providers disagree about whether the base
// carries `/v1`, and guessing a version segment would silently rewrite a URL the user pasted from
// their own dashboard.
export function chatCompletionsUrl(baseUrl) {
  const base = typeof baseUrl === "string" ? baseUrl.trim().replace(/\/+$/, "") : "";
  if (!base) return "";
  return base.endsWith("/chat/completions") ? base : `${base}/chat/completions`;
}

// `GET {base}/models` is part of the same OpenAI-compatible surface. Every hardcoded model list in
// this file is a starting suggestion that ages badly — providers rename and retire models between
// releases — so the settings form can ask the service what it actually serves today.
export function modelsUrl(baseUrl) {
  const base = typeof baseUrl === "string" ? baseUrl.trim().replace(/\/+$/, "") : "";
  if (!base) return "";
  const withoutChat = base.endsWith("/chat/completions") ? base.slice(0, -"/chat/completions".length) : base;
  return withoutChat.endsWith("/models") ? withoutChat : `${withoutChat}/models`;
}

// Providers disagree on the envelope: OpenAI returns `{data:[{id}]}`, some gateways return a bare
// array, and a few return `{models:[...]}`. Ids are de-duplicated and sorted so the datalist is
// stable regardless of upstream ordering.
export function parseModelList(payload) {
  const rows = Array.isArray(payload) ? payload
    : Array.isArray(payload?.data) ? payload.data
      : Array.isArray(payload?.models) ? payload.models
        : [];
  const ids = rows
    .map((row) => (typeof row === "string" ? row : typeof row?.id === "string" ? row.id : typeof row?.name === "string" ? row.name : ""))
    .map((id) => id.trim())
    .filter((id) => id && id.length <= MAXIMUM_MODEL_LENGTH && !CONTROL_CHARACTER_PATTERN.test(id));
  return [...new Set(ids)].sort((first, second) => first.localeCompare(second));
}

const CONTROL_CHARACTER_PATTERN = /[\u0000-\u001f\u007f]/;

function invalid(field, code, message) {
  return { field, code, message };
}

// Header-injection guard: an API key reaching an `Authorization` header must not be able to carry a
// CR/LF and start a second header line.
function validateApiKey(provider, apiKey, errors) {
  const value = typeof apiKey === "string" ? apiKey : "";
  if (!value.trim()) {
    if (provider.requiresApiKey) errors.push(invalid("apiKey", "api_key_missing", "请填写 API Key。"));
    return;
  }
  if (CONTROL_CHARACTER_PATTERN.test(value)) {
    errors.push(invalid("apiKey", "api_key_invalid", "API Key 含有换行或控制字符，请重新复制。"));
    return;
  }
  if (value.length > MAXIMUM_API_KEY_LENGTH) {
    errors.push(invalid("apiKey", "api_key_too_long", `API Key 超过 ${MAXIMUM_API_KEY_LENGTH} 个字符。`));
  }
}

function validateBaseUrl(baseUrl, errors) {
  const value = typeof baseUrl === "string" ? baseUrl.trim() : "";
  if (!value) {
    errors.push(invalid("baseUrl", "base_url_missing", "请填写服务地址。"));
    return;
  }
  if (value.length > MAXIMUM_BASE_URL_LENGTH) {
    errors.push(invalid("baseUrl", "base_url_too_long", `服务地址超过 ${MAXIMUM_BASE_URL_LENGTH} 个字符。`));
    return;
  }
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    errors.push(invalid("baseUrl", "base_url_invalid", "服务地址不是合法 URL，需以 http:// 或 https:// 开头。"));
    return;
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    errors.push(invalid("baseUrl", "base_url_scheme", "服务地址仅支持 http:// 与 https://。"));
    return;
  }
  // A key embedded in the URL would be written to disk in a field the settings form masks
  // elsewhere, and would leak through any error surface that echoes the endpoint.
  if (parsed.username || parsed.password) {
    errors.push(invalid("baseUrl", "base_url_credentials", "服务地址不能内嵌用户名或密码，请改填 API Key。"));
  }
}

function validateModel(model, errors) {
  const value = typeof model === "string" ? model.trim() : "";
  if (!value) {
    errors.push(invalid("model", "model_missing", "请填写模型名称。"));
    return;
  }
  if (value.length > MAXIMUM_MODEL_LENGTH) {
    errors.push(invalid("model", "model_too_long", `模型名称超过 ${MAXIMUM_MODEL_LENGTH} 个字符。`));
    return;
  }
  if (CONTROL_CHARACTER_PATTERN.test(value)) {
    errors.push(invalid("model", "model_invalid", "模型名称含有非法字符。"));
  }
}

function validateStrength(providerId, model, strength, errors) {
  const resolved = resolveStrengthParameter(providerId, model);
  if (resolved.kind === "unsupported") return;
  // Absent means "use this provider's default", which is a legitimate save. Only a value the user
  // actually supplied can be wrong.
  if (strength === undefined || strength === null || strength === "") return;
  if (resolved.kind === "enum") {
    if (!strengthChoiceValues(resolved).includes(strength)) {
      errors.push(invalid("strength", "strength_invalid", `推理强度需为 ${strengthChoiceValues(resolved).join(" / ")} 之一。`));
    }
    return;
  }
  const numeric = typeof strength === "number" ? strength : Number.parseFloat(String(strength));
  if (!Number.isFinite(numeric)) {
    errors.push(invalid("strength", "strength_invalid", "推理强度需为数字。"));
    return;
  }
  if (numeric < resolved.minimum || numeric > resolved.maximum) {
    errors.push(invalid("strength", "strength_out_of_range", `当前服务商的取值范围为 ${resolved.minimum} - ${resolved.maximum}。`));
  }
}

export function defaultAssistantSettings(providerId = DEFAULT_PROVIDER_ID) {
  const provider = providerProfile(providerId) || providerProfile(DEFAULT_PROVIDER_ID);
  const model = provider.models[0] || "";
  const resolved = resolveStrengthParameter(provider.id, model);
  return {
    schemaVersion: ASSISTANT_SETTINGS_SCHEMA_VERSION,
    provider: provider.id,
    baseUrl: provider.baseUrl,
    apiKey: "",
    model,
    strength: resolved.defaultValue,
    personaId: "",
    // Retrieval is not implemented yet; the field is carried so stored settings written by this
    // build survive the knowledge-base stage without a migration.
    knowledgeIds: [],
  };
}

// Coerces any stored or posted shape into the canonical settings object. Never throws: an
// unreadable field falls back to its default so a corrupt file cannot lock the user out of the
// settings form. Validity is a separate question, answered by `validateAssistantSettings`.
export function normalizeAssistantSettings(value) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const provider = providerProfile(source.provider) || providerProfile(DEFAULT_PROVIDER_ID);
  const defaults = defaultAssistantSettings(provider.id);
  const text = (candidate, fallback, limit) => {
    const raw = typeof candidate === "string" ? candidate.trim() : "";
    return raw ? raw.slice(0, limit) : fallback;
  };
  const model = text(source.model, defaults.model, MAXIMUM_MODEL_LENGTH);
  const resolved = resolveStrengthParameter(provider.id, model);
  const strength = resolved.kind === "unsupported"
    ? null
    : clampStrength(resolved, source.strength === undefined ? resolved.defaultValue : source.strength);
  return {
    schemaVersion: ASSISTANT_SETTINGS_SCHEMA_VERSION,
    provider: provider.id,
    baseUrl: text(source.baseUrl, defaults.baseUrl, MAXIMUM_BASE_URL_LENGTH),
    apiKey: typeof source.apiKey === "string" ? source.apiKey.trim().slice(0, MAXIMUM_API_KEY_LENGTH) : "",
    model,
    strength,
    personaId: text(source.personaId, "", 120),
    knowledgeIds: [],
  };
}

// Returns every problem at once. The settings form marks each offending field, so stopping at the
// first error would make the user fix a broken configuration one round trip at a time.
//
// Deliberately inspects the *raw* input rather than the normalized result. Normalisation exists to
// keep a damaged file on disk readable, so it repairs out-of-range and blank values; validating
// after that would silently accept a strength of 99 as the clamped maximum and report success for a
// value the user never chose. `settings` in the result is the normalized form, ready to store once
// the raw input has been judged acceptable.
export function validateAssistantSettings(value) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const settings = normalizeAssistantSettings(source);
  const provider = providerProfile(source.provider === undefined ? settings.provider : source.provider);
  const errors = [];
  if (!provider) {
    errors.push(invalid("provider", "provider_unknown", "未知的服务商。"));
    return { valid: false, errors, settings };
  }
  const rawModel = typeof source.model === "string" ? source.model.trim() : "";
  validateBaseUrl(source.baseUrl, errors);
  validateApiKey(provider, source.apiKey, errors);
  validateModel(source.model, errors);
  validateStrength(provider.id, rawModel || settings.model, source.strength, errors);
  return { valid: errors.length === 0, errors, settings };
}

// What the control plane is allowed to hand back to the browser: identical to the stored settings
// except the secret is replaced by a presence flag and a masked tail for recognition.
export function redactAssistantSettings(settings) {
  const normalized = normalizeAssistantSettings(settings);
  const key = normalized.apiKey;
  const { apiKey, ...rest } = normalized;
  return {
    ...rest,
    hasApiKey: Boolean(key),
    apiKeyHint: key ? `••••${key.slice(-4)}` : "",
  };
}
