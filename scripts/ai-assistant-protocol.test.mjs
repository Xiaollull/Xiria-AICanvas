import assert from "node:assert/strict";
import test from "node:test";
import {
  MAXIMUM_HISTORY_MESSAGES,
  PROMPT_PROTOCOL_INSTRUCTION,
  buildChatMessages,
  buildChatRequestBody,
  composeSystemPrompt,
  conversationMessages,
  createStreamDecoder,
  extractPromptBlocks,
  latestPromptSuggestion,
  normalizeConversation,
  optimizationRequest,
  providerErrorMessage,
  readPromptSnapshot,
  segmentAssistantReply,
  writePromptSnapshot,
  DEFAULT_SESSION_TITLE,
  MAXIMUM_SESSION_TITLE,
  SESSION_SCHEMA_VERSION,
  deriveSessionTitle,
  normalizeSession,
  sessionSummary,
  sortSessions,
  validSessionId,
} from "../src/ai-assistant-protocol.js";

const memoryStorage = () => {
  const store = new Map();
  return {
    getItem: (key) => (store.has(key) ? store.get(key) : null),
    setItem: (key, value) => { store.set(key, String(value)); },
  };
};

test("a frame split across network chunks is buffered, not parsed short", () => {
  const decoder = createStreamDecoder();
  assert.deepEqual(decoder.push('data: {"choices":[{"delta":{"content":"你'), []);
  const events = decoder.push('好"}}]}\n\n');
  assert.equal(events.length, 1);
  assert.equal(events[0].content, "你好");
});

test("comments, blank frames and CRLF terminators are handled", () => {
  const decoder = createStreamDecoder();
  const events = decoder.push(': keepalive\r\n\r\ndata: {"choices":[{"delta":{"content":"a"}}]}\r\n\r\ndata: [DONE]\r\n\r\n');
  assert.deepEqual(events.map((event) => event.type), ["delta", "done"]);
});

test("reasoning content is carried separately from answer content", () => {
  const decoder = createStreamDecoder();
  const [event] = decoder.push('data: {"choices":[{"delta":{"reasoning_content":"想一想"}}]}\n\n');
  assert.equal(event.reasoning, "想一想");
  // Chain-of-thought must never be mistaken for the answer, or it would end up inside a prompt.
  assert.equal(event.content, "");
});

test("a mid-stream error frame is surfaced as an error event", () => {
  const decoder = createStreamDecoder();
  const [event] = decoder.push('data: {"error":{"message":"rate limited"}}\n\n');
  assert.equal(event.type, "error");
  assert.equal(event.message, "rate limited");
});

test("unparseable frames are dropped rather than throwing mid-stream", () => {
  const decoder = createStreamDecoder();
  assert.deepEqual(decoder.push("data: {oops\n\n"), []);
  assert.deepEqual(decoder.push("data: \n\n"), []);
  const [event] = decoder.push('data: {"choices":[{"delta":{"content":"ok"}}]}\n\n');
  assert.equal(event.content, "ok", "the stream keeps working after a bad frame");
});

test("flush emits a trailing frame that never received its terminator", () => {
  const decoder = createStreamDecoder();
  assert.deepEqual(decoder.push('data: {"choices":[{"delta":{"content":"tail"}}]}'), []);
  assert.equal(decoder.flush()[0].content, "tail");
  assert.deepEqual(decoder.flush(), []);
});

test("only closed fences yield an applicable prompt", () => {
  assert.equal(latestPromptSuggestion("```prompt\nhalf written"), null);
  assert.deepEqual(latestPromptSuggestion("```prompt\n1girl\n```"), { positive: "1girl", negative: "" });
  assert.equal(extractPromptBlocks("no fences here").length, 0);
  assert.equal(extractPromptBlocks("```prompt\n\n```").length, 0, "an empty fence is not a prompt");
});

test("the last block of each kind wins so a self-revising reply applies its final answer", () => {
  const reply = "初稿\n```prompt\nfirst\n```\n修订\n```prompt\nsecond\n```\n```negative-prompt\nlowres\n```";
  assert.deepEqual(latestPromptSuggestion(reply), { positive: "second", negative: "lowres" });
});

test("replies split into prose and applicable prompt cards in order", () => {
  const segments = segmentAssistantReply("说明\n```prompt\ntags\n```\n补充\n```negative\nbad\n```");
  assert.deepEqual(segments, [
    { type: "text", content: "说明" },
    { type: "positive", content: "tags" },
    { type: "text", content: "补充" },
    { type: "negative", content: "bad" },
  ]);
  assert.deepEqual(segmentAssistantReply("只有文字"), [{ type: "text", content: "只有文字" }]);
  assert.deepEqual(segmentAssistantReply(""), []);
});

test("the output protocol is appended to every persona, including none at all", () => {
  assert.ok(composeSystemPrompt({ systemPrompt: "你是画师" }).startsWith("你是画师"));
  assert.ok(composeSystemPrompt({ systemPrompt: "你是画师" }).includes(PROMPT_PROTOCOL_INSTRUCTION));
  // Without it the model has no reason to fence anything and the apply button could never appear.
  assert.ok(composeSystemPrompt(null).includes(PROMPT_PROTOCOL_INSTRUCTION));
});

test("knowledge entries render into the reserved section when they eventually exist", () => {
  const withKnowledge = composeSystemPrompt({ systemPrompt: "角色" }, [{ content: "片段A" }, { content: "" }]);
  assert.ok(withKnowledge.includes("【知识库】"));
  assert.ok(withKnowledge.includes("片段A"));
  assert.ok(!composeSystemPrompt({ systemPrompt: "角色" }, []).includes("【知识库】"));
});

test("in-flight and failed turns are never replayed as context", () => {
  const history = [
    { role: "user", content: "一" },
    { role: "assistant", content: "半截", streaming: true },
    { role: "assistant", content: "出错了", failed: true },
    { role: "system", content: "注入" },
    { role: "assistant", content: "  " },
    { role: "assistant", content: "二" },
  ];
  assert.deepEqual(conversationMessages(history), [{ role: "user", content: "一" }, { role: "assistant", content: "二" }]);
});

test("history is capped so a long session cannot grow the request without bound", () => {
  const history = Array.from({ length: MAXIMUM_HISTORY_MESSAGES + 20 }, (_, index) => ({ role: "user", content: `m${index}` }));
  const messages = conversationMessages(history);
  assert.equal(messages.length, MAXIMUM_HISTORY_MESSAGES);
  assert.equal(messages.at(-1).content, `m${history.length - 1}`, "the newest turns are the ones kept");
});

test("the assembled request carries the system message first", () => {
  const messages = buildChatMessages({ persona: { systemPrompt: "角色" }, history: [{ role: "user", content: "hi" }] });
  assert.equal(messages[0].role, "system");
  assert.deepEqual(messages[1], { role: "user", content: "hi" });
  const body = buildChatRequestBody({ messages, model: "deepseek-chat", strength: { temperature: 1.3 } });
  assert.deepEqual(body, { model: "deepseek-chat", messages, stream: true, temperature: 1.3 });
  // An unsupported strength contributes no key at all.
  assert.equal("temperature" in buildChatRequestBody({ messages, model: "x", strength: {} }), false);
});

test("the one-shot optimisation request preserves the old button's behaviour", () => {
  const request = optimizationRequest("a cat", "lowres");
  assert.ok(request.includes("a cat"));
  assert.ok(request.includes("lowres"));
  assert.ok(request.includes("```prompt"), "the reply must be fenced so it can be applied directly");
  assert.ok(!optimizationRequest("a cat", "").includes("负向提示词"));
  assert.ok(optimizationRequest("", "").includes("(空)"));
});

test("provider error envelopes and bare status codes both produce a readable sentence", () => {
  assert.equal(providerErrorMessage({ error: { message: "Authentication Fails" } }), "Authentication Fails");
  assert.equal(providerErrorMessage({ error: "flat string" }), "flat string");
  assert.equal(providerErrorMessage({ detail: "fastapi style" }), "fastapi style");
  assert.match(providerErrorMessage(null, 401), /API Key/);
  assert.match(providerErrorMessage(null, 404), /v1/);
  assert.match(providerErrorMessage(null, 429), /限流/);
  assert.match(providerErrorMessage(null, 500), /HTTP 500/);
  assert.equal(typeof providerErrorMessage(null), "string");
});

test("stored conversations survive a reload and reject foreign roles", () => {
  const normalized = normalizeConversation({
    messages: [
      { id: "a", role: "user", content: "一" },
      { role: "system", content: "注入" },
      { role: "assistant", content: "二", reasoning: "思考", failed: true },
      { role: "assistant", content: 42 },
    ],
  });
  assert.deepEqual(normalized.messages.map((message) => message.role), ["user", "assistant"]);
  assert.equal(normalized.messages[1].failed, true);
  assert.equal(normalized.messages[1].reasoning, "思考");
  assert.ok(normalized.messages.every((message) => typeof message.id === "string" && message.id));
  assert.deepEqual(normalizeConversation(null).messages, []);
});

test("the prompt snapshot round-trips for the popped-out tab", () => {
  const storage = memoryStorage();
  assert.deepEqual(readPromptSnapshot(storage), { positive: "", negative: "" });
  writePromptSnapshot(storage, { positive: "a cat", negative: "lowres" });
  assert.deepEqual(readPromptSnapshot(storage), { positive: "a cat", negative: "lowres" });
  writePromptSnapshot(storage, { positive: null, negative: undefined });
  assert.deepEqual(readPromptSnapshot(storage), { positive: "", negative: "" });
  // A storage that throws must not take the caller down with it.
  assert.doesNotThrow(() => writePromptSnapshot({ setItem() { throw new Error("quota"); } }, { positive: "x" }));
  assert.deepEqual(readPromptSnapshot({ getItem() { throw new Error("blocked"); } }), { positive: "", negative: "" });
});

test("session ids accept only the generated shape, so an id can never become a path", () => {
  assert.equal(validSessionId("3f4a2b1c-1111-2222-3333-444455556666"), true);
  for (const hostile of ["../../etc/passwd", "..", "a/b", "3f4a2b1c-1111-2222-3333-44445555666", "", null, "3F4A2B1C-1111-2222-3333-444455556666"]) {
    assert.equal(validSessionId(hostile), false, `${String(hostile)} must be refused`);
  }
});

test("session titles are derived from the first real user line", () => {
  assert.equal(deriveSessionTitle([{ role: "user", content: "画一只赛博朋克风格的猫" }]), "画一只赛博朋克风格的猫");
  // Fenced blocks and the optimisation preamble make useless titles.
  assert.equal(deriveSessionTitle([{ role: "user", content: "```prompt\n1girl\n```\n【当前正向提示词】\n改成夜景" }]), "改成夜景");
  assert.equal(deriveSessionTitle([{ role: "assistant", content: "我先说" }]), DEFAULT_SESSION_TITLE);
  assert.equal(deriveSessionTitle([]), DEFAULT_SESSION_TITLE);
  assert.equal(deriveSessionTitle([{ role: "user", content: "   " }]), DEFAULT_SESSION_TITLE);
  const long = deriveSessionTitle([{ role: "user", content: "长".repeat(200) }]);
  assert.ok(long.length <= MAXIMUM_SESSION_TITLE);
  assert.ok(long.endsWith("…"));
});

test("sessions normalise without throwing and keep their creation time", () => {
  const id = "3f4a2b1c-1111-2222-3333-444455556666";
  const session = normalizeSession({ createdAt: "2026-01-01T00:00:00.000Z", messages: [{ role: "user", content: "你好" }] }, { id });
  assert.equal(session.id, id);
  assert.equal(session.title, "你好");
  assert.equal(session.createdAt, "2026-01-01T00:00:00.000Z");
  assert.equal(session.schemaVersion, SESSION_SCHEMA_VERSION);
  // A damaged file still opens, otherwise its history would be permanently unreachable.
  assert.doesNotThrow(() => normalizeSession("garbage", { id }));
  assert.deepEqual(normalizeSession(null, { id }).messages, []);
  assert.equal(normalizeSession({ id: "../evil" }, {}).id, "", "a bad stored id is dropped, not trusted");
  // An explicit title survives; a blank one is re-derived.
  assert.equal(normalizeSession({ title: "我的对话", messages: [{ role: "user", content: "你好" }] }, { id }).title, "我的对话");
});

test("sessions sort by most recent activity, which is also the prune order", () => {
  const sorted = sortSessions([
    { id: "old", updatedAt: "2026-01-01T00:00:00.000Z" },
    { id: "new", updatedAt: "2026-08-01T00:00:00.000Z" },
    { id: "mid", updatedAt: "2026-04-01T00:00:00.000Z" },
  ]);
  assert.deepEqual(sorted.map((entry) => entry.id), ["new", "mid", "old"]);
});

test("session summaries carry metadata only, never the transcript", () => {
  const session = normalizeSession({
    messages: [{ role: "user", content: "第一句" }, { role: "assistant", content: "机密回复内容" }],
  }, { id: "3f4a2b1c-1111-2222-3333-444455556666" });
  const summary = sessionSummary(session);
  assert.deepEqual(Object.keys(summary).sort(), ["createdAt", "id", "messageCount", "title", "updatedAt"]);
  assert.equal(summary.messageCount, 2);
  assert.equal(summary.messages, undefined, "the picker must not receive the transcript");
  // The title is deliberately derived from the first user line, so only the rest of the
  // conversation has to stay out of a listing.
  assert.equal(summary.title, "第一句");
  assert.ok(!JSON.stringify(summary).includes("机密回复内容"), "listing must not ship message bodies");
});
