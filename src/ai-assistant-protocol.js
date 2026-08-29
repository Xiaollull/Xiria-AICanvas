// Wire protocol for the AI assistant: streamed response decoding, conversation shaping, and the
// fenced-block contract that lets a chat reply carry a prompt the workspace can apply.
//
// Kept free of React and of Node built-ins so the browser bundle, the control plane, and the unit
// tests all exercise the same parsing code.

import { personaSections } from "./assistant-persona.js";

export const ASSISTANT_CONVERSATION_SCHEMA_VERSION = 1;

// Cross-tab bus. The popped-out `/assistant` tab and the workspace tab are separate documents, so
// applying a prompt from the pop-out has to travel over BroadcastChannel rather than a callback.
export const ASSISTANT_CHANNEL_NAME = "xirai-assistant-v1";
export const ASSISTANT_APPLY_PROMPT = "apply-prompt";
export const ASSISTANT_CONVERSATION_SYNC = "conversation-sync";
// Broadcast after any session write so a second surface refetches instead of showing stale history.
export const ASSISTANT_SESSIONS_CHANGED = "sessions-changed";
// Which session each surface is looking at. Shared so popping out continues the same conversation.
export const ASSISTANT_ACTIVE_SESSION_KEY = "xirai-assistant-active-session-v1";

// The workspace mirrors its prompt boxes here so the popped-out tab, which has no prompt boxes of
// its own, can still run "optimise current prompt" against what the user is actually looking at.
export const ASSISTANT_PROMPT_SNAPSHOT_KEY = "xirai-assistant-prompt-snapshot-v1";

export function readPromptSnapshot(storage) {
  try {
    const saved = JSON.parse(storage?.getItem(ASSISTANT_PROMPT_SNAPSHOT_KEY) || "null");
    return {
      positive: typeof saved?.positive === "string" ? saved.positive : "",
      negative: typeof saved?.negative === "string" ? saved.negative : "",
    };
  } catch {
    return { positive: "", negative: "" };
  }
}

export function writePromptSnapshot(storage, snapshot) {
  try {
    storage?.setItem(ASSISTANT_PROMPT_SNAPSHOT_KEY, JSON.stringify({
      positive: clampText(snapshot?.positive),
      negative: clampText(snapshot?.negative),
    }));
  } catch {
    // Storage quota failures must not interfere with typing in the prompt box.
  }
}

export const MAXIMUM_HISTORY_MESSAGES = 24;
export const MAXIMUM_MESSAGE_CHARACTERS = 8000;
export const MAXIMUM_STORED_MESSAGES = 200;

// Chat sessions. Conversations are kept server-side, one file each, so "previous chats" survive a
// cleared browser profile and both the floating window and the popped-out tab read one source of
// truth instead of two diverging copies.
export const SESSION_SCHEMA_VERSION = 1;
export const MAXIMUM_SESSIONS = 50;
export const MAXIMUM_SESSION_TITLE = 60;
export const DEFAULT_SESSION_TITLE = "新对话";

// Ids are generated server-side and appear in a URL path that becomes a filename, so the accepted
// shape is deliberately narrow rather than "whatever the client sent".
export const SESSION_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

export function validSessionId(value) {
  return typeof value === "string" && SESSION_ID_PATTERN.test(value);
}

// The model is asked to fence any prompt it wants applied. Free prose can then surround it without
// the workspace guessing which sentence was the deliverable.
export const POSITIVE_FENCE = "prompt";
export const NEGATIVE_FENCE = "negative";

const FENCE_PATTERN = /```(prompt|negative(?:-prompt)?)[^\S\r\n]*\r?\n([\s\S]*?)```/gi;

// Appended to every persona. Without it a provider has no idea the caller can act on its output,
// and the apply button would never light up.
export const PROMPT_PROTOCOL_INSTRUCTION = [
  "【输出协议】当你给出可直接使用的绘画提示词时，必须放进围栏代码块：",
  "正向提示词用 ```prompt 围栏，负向提示词用 ```negative 围栏。",
  "围栏内只写提示词本身，使用英文逗号分隔的标签，不要写解释、不要写编号、不要再嵌套围栏。",
  "围栏之外可以正常用中文说明你的修改思路。若本轮只是讨论而没有产出提示词，则不要输出围栏。",
].join("\n");

export function composeSystemPrompt(persona, knowledge = []) {
  // Identity, prose, specialties and behaviour rules, in that fixed order. A section renders only
  // when it has content, so a prose-only character composes exactly as it did before characters
  // had attributes, and `description` is never included — it is picker metadata, written for
  // whoever is choosing a character rather than for the model.
  const sections = [...personaSections(persona)];
  // Retrieval is not implemented yet. Entries are accepted and rendered so the knowledge-base stage
  // only has to fill this array, not restructure the system message.
  const snippets = Array.isArray(knowledge)
    ? knowledge.map((entry) => (typeof entry?.content === "string" ? entry.content.trim() : "")).filter(Boolean)
    : [];
  if (snippets.length) sections.push(`【知识库】\n${snippets.join("\n\n")}`);
  sections.push(PROMPT_PROTOCOL_INSTRUCTION);
  return sections.join("\n\n");
}

function clampText(value) {
  const text = typeof value === "string" ? value : "";
  return text.length > MAXIMUM_MESSAGE_CHARACTERS ? text.slice(0, MAXIMUM_MESSAGE_CHARACTERS) : text;
}

// Storage shape -> provider shape. Drops anything that is not a completed user/assistant turn:
// a half-streamed reply or an error bubble must never be replayed as context.
export function conversationMessages(history) {
  const entries = Array.isArray(history) ? history : [];
  return entries
    .filter((entry) => (entry?.role === "user" || entry?.role === "assistant")
      && !entry.failed
      && !entry.streaming
      && typeof entry.content === "string"
      && entry.content.trim())
    .slice(-MAXIMUM_HISTORY_MESSAGES)
    .map((entry) => ({ role: entry.role, content: clampText(entry.content) }));
}

export function buildChatMessages({ persona, history, knowledge } = {}) {
  const system = composeSystemPrompt(persona, knowledge);
  const messages = conversationMessages(history);
  return system ? [{ role: "system", content: system }, ...messages] : messages;
}

// The one-shot path behind the old optimise action: no conversation required, and the reply is
// constrained to a single fenced block so it can be applied without the user reading it.
export function optimizationRequest(positive, negative = "") {
  const lines = ["请优化下面的正向提示词，保留原有主体与风格意图，补充画质、光照、构图与细节描述。"];
  lines.push("直接给出结果，用 ```prompt 围栏包裹，围栏外只写一句话说明改动重点。");
  lines.push("", "【当前正向提示词】", positive.trim() || "(空)");
  const trimmedNegative = typeof negative === "string" ? negative.trim() : "";
  if (trimmedNegative) lines.push("", "【当前负向提示词】", trimmedNegative);
  return lines.join("\n");
}

// Only closed fences match, so a block still streaming in cannot be applied half-written.
export function extractPromptBlocks(text) {
  const source = typeof text === "string" ? text : "";
  const blocks = [];
  FENCE_PATTERN.lastIndex = 0;
  let match = FENCE_PATTERN.exec(source);
  while (match) {
    const content = match[2].trim();
    if (content) blocks.push({ kind: match[1].toLowerCase().startsWith("negative") ? "negative" : "positive", content });
    match = FENCE_PATTERN.exec(source);
  }
  return blocks;
}

// Splits a reply into prose and fenced prompts so the transcript can render each prompt as an
// applicable card instead of as raw markdown the user would have to select and copy.
export function segmentAssistantReply(text) {
  const source = typeof text === "string" ? text : "";
  const segments = [];
  let cursor = 0;
  FENCE_PATTERN.lastIndex = 0;
  let match = FENCE_PATTERN.exec(source);
  while (match) {
    const lead = source.slice(cursor, match.index);
    if (lead.trim()) segments.push({ type: "text", content: lead.trim() });
    const content = match[2].trim();
    if (content) segments.push({ type: match[1].toLowerCase().startsWith("negative") ? "negative" : "positive", content });
    cursor = match.index + match[0].length;
    match = FENCE_PATTERN.exec(source);
  }
  const tail = source.slice(cursor);
  if (tail.trim()) segments.push({ type: "text", content: tail.trim() });
  return segments;
}

// Last block of each kind wins: a reply that revises itself should apply its final answer.
export function latestPromptSuggestion(text) {
  const blocks = extractPromptBlocks(text);
  if (!blocks.length) return null;
  const positive = blocks.filter((block) => block.kind === "positive").at(-1)?.content || "";
  const negative = blocks.filter((block) => block.kind === "negative").at(-1)?.content || "";
  return positive || negative ? { positive, negative } : null;
}

export function buildChatRequestBody({ messages, model, strength, stream = true }) {
  return { model, messages, stream, ...(strength || {}) };
}

function parseFrame(frame) {
  const dataLines = [];
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith(":")) continue;
    if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
  }
  if (!dataLines.length) return null;
  const payload = dataLines.join("\n");
  if (payload === "[DONE]") return { type: "done" };
  let parsed;
  try {
    parsed = JSON.parse(payload);
  } catch {
    return null;
  }
  // Some gateways report a mid-stream failure as a normal SSE frame carrying an error object
  // instead of closing with a non-200, so the stream reader has to recognise it.
  if (parsed?.error) {
    return { type: "error", message: providerErrorMessage(parsed) };
  }
  const delta = parsed?.choices?.[0]?.delta || {};
  const content = typeof delta.content === "string" ? delta.content : "";
  // DeepSeek's reasoner streams its chain of thought on a separate key; it is shown in a collapsed
  // panel and deliberately excluded from `content` so it never leaks into an applied prompt.
  const reasoning = typeof delta.reasoning_content === "string" ? delta.reasoning_content : "";
  const finish = parsed?.choices?.[0]?.finish_reason || null;
  if (!content && !reasoning && !finish) return null;
  return { type: "delta", content, reasoning, finish };
}

// Incremental SSE reader. A network chunk can split a frame anywhere, including inside a UTF-8
// sequence or between `data:` and its payload, so unterminated bytes stay buffered until the next
// push rather than being parsed as a short frame.
export function createStreamDecoder() {
  let buffer = "";
  const drain = (flush) => {
    const events = [];
    for (;;) {
      const match = /\r?\n\r?\n/.exec(buffer);
      if (!match) break;
      const frame = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);
      const event = parseFrame(frame);
      if (event) events.push(event);
    }
    if (flush && buffer.trim()) {
      const event = parseFrame(buffer);
      buffer = "";
      if (event) events.push(event);
    }
    return events;
  };
  return {
    push(chunk) {
      buffer += typeof chunk === "string" ? chunk : "";
      return drain(false);
    },
    flush() {
      return drain(true);
    },
  };
}

// Providers disagree on error envelopes; this reaches for the human-readable field each uses before
// falling back to the raw shape.
export function providerErrorMessage(payload, status = 0) {
  const error = payload?.error;
  const candidates = [
    typeof error === "string" ? error : "",
    typeof error?.message === "string" ? error.message : "",
    typeof payload?.message === "string" ? payload.message : "",
    typeof payload?.detail === "string" ? payload.detail : "",
  ];
  const message = candidates.find((candidate) => candidate && candidate.trim());
  if (message) return message.trim().slice(0, 400);
  if (status === 401 || status === 403) return "服务端拒绝了这个 API Key（401/403），请检查密钥与账户权限。";
  if (status === 404) return "服务地址或模型名称不存在（404），请检查 URL 是否需要 /v1 后缀。";
  if (status === 429) return "触发服务商限流（429），请稍后再试。";
  if (status) return `服务返回 HTTP ${status}。`;
  return "服务返回了无法解析的响应。";
}

export function normalizeConversation(value) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const messages = Array.isArray(source.messages) ? source.messages : [];
  return {
    schemaVersion: ASSISTANT_CONVERSATION_SCHEMA_VERSION,
    messages: messages
      .filter((entry) => (entry?.role === "user" || entry?.role === "assistant") && typeof entry.content === "string")
      .slice(-MAXIMUM_STORED_MESSAGES)
      .map((entry) => ({
        id: typeof entry.id === "string" && entry.id ? entry.id : `${entry.role}-${Math.random().toString(36).slice(2)}`,
        role: entry.role,
        content: clampText(entry.content),
        reasoning: typeof entry.reasoning === "string" ? clampText(entry.reasoning) : "",
        failed: entry.failed === true,
      })),
  };
}

// A session's title is derived rather than asked for: the picker has to be readable immediately,
// and prompting for a name before the first message would be friction for a throwaway chat.
export function deriveSessionTitle(messages) {
  const first = (Array.isArray(messages) ? messages : []).find((message) => message?.role === "user" && typeof message.content === "string" && message.content.trim());
  if (!first) return DEFAULT_SESSION_TITLE;
  // Fenced prompt blocks and the one-shot optimisation preamble make poor titles, so the derivation
  // reads the first ordinary line instead of the raw head of the message.
  const line = first.content
    .replace(/```[\s\S]*?```/g, " ")
    .split(/\r?\n/)
    .map((candidate) => candidate.trim())
    .find((candidate) => candidate && !candidate.startsWith("【")) || "";
  const title = line.replace(/\s+/g, " ").trim();
  if (!title) return DEFAULT_SESSION_TITLE;
  return title.length > MAXIMUM_SESSION_TITLE ? `${title.slice(0, MAXIMUM_SESSION_TITLE - 1)}…` : title;
}

function isoTimestamp(value, fallback) {
  if (typeof value === "string") {
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) return new Date(parsed).toISOString();
  }
  return fallback;
}

// Never throws: a session file damaged on disk still has to open, otherwise one bad write would
// make its history permanently unreachable.
export function normalizeSession(value, { id, now = new Date().toISOString() } = {}) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const sessionId = validSessionId(id) ? id : (validSessionId(source.id) ? source.id : "");
  const messages = normalizeConversation(source).messages;
  const explicit = typeof source.title === "string" ? source.title.trim().slice(0, MAXIMUM_SESSION_TITLE) : "";
  const createdAt = isoTimestamp(source.createdAt, now);
  return {
    schemaVersion: SESSION_SCHEMA_VERSION,
    id: sessionId,
    title: explicit || deriveSessionTitle(messages),
    createdAt,
    updatedAt: isoTimestamp(source.updatedAt, createdAt),
    messages,
  };
}

// The picker only needs metadata. Listing full transcripts would send every stored conversation to
// the browser just to render a sidebar.
export function sessionSummary(session) {
  return {
    id: session.id,
    title: session.title,
    createdAt: session.createdAt,
    updatedAt: session.updatedAt,
    messageCount: session.messages.length,
  };
}

// Newest activity first, which is the order the picker shows and the order the cap trims from.
export function sortSessions(sessions) {
  return [...sessions].sort((first, second) => String(second.updatedAt).localeCompare(String(first.updatedAt)));
}
