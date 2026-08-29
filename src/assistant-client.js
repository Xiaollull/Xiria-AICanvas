// Browser side of the assistant transport. Every call goes to the local control plane, never to a
// provider: the API key stays on disk and the browser never holds it.

import { createStreamDecoder } from "./ai-assistant-protocol.js";

const JSON_HEADERS = { "Content-Type": "application/json" };

async function readError(response, fallback) {
  try {
    const payload = await response.json();
    return { message: payload.error || fallback, errors: Array.isArray(payload.errors) ? payload.errors : [] };
  } catch {
    return { message: fallback, errors: [] };
  }
}

export class AssistantRequestError extends Error {
  constructor(message, errors = []) {
    super(message);
    this.name = "AssistantRequestError";
    // Per-field codes from the validator, so the settings form can mark the offending inputs.
    this.errors = errors;
  }
}

export async function fetchAssistantSettings(signal) {
  const response = await fetch("/api/assistant/settings", { cache: "no-store", signal });
  if (!response.ok) {
    const { message, errors } = await readError(response, "无法读取 AI 助手配置");
    throw new AssistantRequestError(message, errors);
  }
  return response.json();
}

export async function saveAssistantSettings(settings, signal) {
  const response = await fetch("/api/assistant/settings", {
    method: "PUT",
    headers: JSON_HEADERS,
    body: JSON.stringify(settings),
    signal,
  });
  if (!response.ok) {
    const { message, errors } = await readError(response, "保存失败");
    throw new AssistantRequestError(message, errors);
  }
  return response.json();
}

// Library management — service configurations and characters. Every one of these resolves with the
// refreshed collection, because each page draws its list and its editor from the same state and a
// per-row reply would cost a second round trip before either could redraw.
async function libraryRequest(target, { method = "GET", body, fallback, signal } = {}) {
  const response = await fetch(target, {
    method,
    cache: "no-store",
    ...(body === undefined ? {} : { headers: JSON_HEADERS, body: JSON.stringify(body) }),
    signal,
  });
  if (!response.ok) {
    const { message, errors } = await readError(response, fallback);
    throw new AssistantRequestError(message, errors);
  }
  return response.json();
}

export function fetchAssistantProfiles(signal) {
  return libraryRequest("/api/assistant/profiles", { fallback: "无法读取 AI 助手配置", signal });
}

export function createAssistantProfile(seed, signal) {
  return libraryRequest("/api/assistant/profiles", { method: "POST", body: seed || {}, fallback: "无法新建配置", signal });
}

export function saveAssistantProfile(id, patch, signal) {
  return libraryRequest(`/api/assistant/profiles/${encodeURIComponent(id)}`, {
    method: "PUT", body: patch, fallback: "保存失败", signal,
  });
}

export function deleteAssistantProfile(id, signal) {
  return libraryRequest(`/api/assistant/profiles/${encodeURIComponent(id)}`, {
    method: "DELETE", fallback: "无法删除该配置", signal,
  });
}

export function activateAssistantProfile(id, signal) {
  return libraryRequest(`/api/assistant/profiles/${encodeURIComponent(id)}/activate`, {
    method: "POST", body: {}, fallback: "无法切换配置", signal,
  });
}

export function duplicateAssistantProfile(id, signal) {
  return libraryRequest(`/api/assistant/profiles/${encodeURIComponent(id)}/duplicate`, {
    method: "POST", body: {}, fallback: "无法复制该配置", signal,
  });
}

export async function fetchAssistantPersonas(signal) {
  const response = await fetch("/api/assistant/personas", { cache: "no-store", signal });
  if (!response.ok) throw new AssistantRequestError("无法读取角色设定");
  return response.json();
}

// The character library. Like the profile calls, each of these resolves with the refreshed library
// so the interface redraws its list and its editor from one reply. Built-in characters have no
// stored file, so editing or deleting one answers 409 with a `persona_readonly` code.
export function createAssistantPersona(seed, signal) {
  return libraryRequest("/api/assistant/personas", { method: "POST", body: seed || {}, fallback: "无法新建角色", signal });
}

export function saveAssistantPersona(id, persona, signal) {
  return libraryRequest(`/api/assistant/personas/${encodeURIComponent(id)}`, {
    method: "PUT", body: persona, fallback: "无法保存角色", signal,
  });
}

export function deleteAssistantPersona(id, signal) {
  return libraryRequest(`/api/assistant/personas/${encodeURIComponent(id)}`, {
    method: "DELETE", fallback: "无法删除该角色", signal,
  });
}

// Asks the configured service what it actually serves today. The bundled per-provider lists are
// only a starting point; providers rename and retire models between releases.
export async function fetchAssistantModels(signal) {
  const response = await fetch("/api/assistant/models", { method: "POST", headers: JSON_HEADERS, body: "{}", signal });
  if (!response.ok) {
    const { message, errors } = await readError(response, "无法获取模型列表");
    throw new AssistantRequestError(message, errors);
  }
  return (await response.json()).models || [];
}

export async function fetchAssistantSessions(signal) {
  const response = await fetch("/api/assistant/sessions", { cache: "no-store", signal });
  if (!response.ok) throw new AssistantRequestError("无法读取历史对话");
  return (await response.json()).sessions || [];
}

export async function createAssistantSession(seed, signal) {
  const response = await fetch("/api/assistant/sessions", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(seed || {}),
    signal,
  });
  if (!response.ok) throw new AssistantRequestError("无法新建对话");
  return (await response.json()).session;
}

export async function fetchAssistantSession(id, signal) {
  const response = await fetch(`/api/assistant/sessions/${encodeURIComponent(id)}`, { cache: "no-store", signal });
  // A missing session is an ordinary outcome — another tab may have deleted it — so the caller
  // gets null and starts a new one rather than an error bubble.
  if (response.status === 404) return null;
  if (!response.ok) throw new AssistantRequestError("无法读取该对话");
  return (await response.json()).session;
}

export async function saveAssistantSession(id, session, signal) {
  const response = await fetch(`/api/assistant/sessions/${encodeURIComponent(id)}`, {
    method: "PUT",
    headers: JSON_HEADERS,
    body: JSON.stringify(session),
    signal,
  });
  if (response.status === 404) return null;
  if (!response.ok) throw new AssistantRequestError("无法保存对话");
  return (await response.json()).session;
}

export async function deleteAssistantSession(id, signal) {
  const response = await fetch(`/api/assistant/sessions/${encodeURIComponent(id)}`, { method: "DELETE", signal });
  if (!response.ok) throw new AssistantRequestError("无法删除该对话");
  return true;
}

export async function testAssistantConnection(signal) {
  const response = await fetch("/api/assistant/test", { method: "POST", headers: JSON_HEADERS, body: "{}", signal });
  if (!response.ok) {
    const { message, errors } = await readError(response, "连接测试失败");
    throw new AssistantRequestError(message, errors);
  }
  return response.json();
}

// Streams a reply, invoking `onDelta` for each fragment. Resolves with the finished text once the
// provider closes the stream; rejects with an actionable message on any failure, including a
// mid-stream error frame, so the caller can render one failed bubble rather than a silent stall.
export async function streamAssistantChat({ history, onDelta, signal }) {
  const response = await fetch("/api/assistant/chat", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ history }),
    signal,
  });
  if (!response.ok) {
    const { message, errors } = await readError(response, "AI 服务请求失败");
    throw new AssistantRequestError(message, errors);
  }
  if (!response.body) throw new AssistantRequestError("当前浏览器不支持流式响应");
  const reader = response.body.getReader();
  const textDecoder = new TextDecoder();
  const sse = createStreamDecoder();
  let content = "";
  let reasoning = "";
  const consume = (events) => {
    for (const event of events) {
      if (event.type === "error") throw new AssistantRequestError(event.message);
      if (event.type !== "delta") continue;
      content += event.content;
      reasoning += event.reasoning;
      onDelta?.({ content, reasoning, chunk: event.content });
    }
  };
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      consume(sse.push(textDecoder.decode(value, { stream: true })));
    }
    consume(sse.flush());
  } finally {
    reader.releaseLock?.();
  }
  return { content, reasoning };
}
