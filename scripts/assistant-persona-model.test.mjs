import assert from "node:assert/strict";
import test from "node:test";

import { composeSystemPrompt, PROMPT_PROTOCOL_INSTRUCTION } from "../src/ai-assistant-protocol.js";
import {
  MAXIMUM_PERSONA_RULES,
  MAXIMUM_PERSONA_SPECIALTIES,
  MAXIMUM_PERSONA_STARTERS,
  PERSONA_ATTRIBUTES,
  PERSONA_SCHEMA_VERSION,
  draftAssistantPersona,
  hasPersonaAttributes,
  normalizeAssistantPersona,
  personaSections,
  sortAssistantPersonas,
  uniquePersonaName,
  validPersonaId,
  validUserPersonaId,
  validateAssistantPersona,
} from "../src/assistant-persona.js";

const UUID = "3f4a2b1c-1111-2222-3333-444455556666";

const character = (overrides = {}) => ({
  id: UUID,
  name: "赛博朋克摄影师",
  description: "偏爱霓虹夜景与胶片颗粒的摄影指导",
  traits: { tone: "professional", detail: "balanced", language: "zh", initiative: "ask-first" },
  specialties: ["电影感人像", "赛博朋克夜景"],
  rules: ["先给提示词，再说明改动重点", "不要堆砌无意义的画质词"],
  systemPrompt: "你是一位擅长霓虹夜景的摄影指导。",
  starters: ["帮我写一段雨后街道的提示词"],
  ...overrides,
});

test("a prose-only character composes exactly as it did before characters had attributes", () => {
  // Every shipped character is one of these, and this is the property that let attributes be added
  // without changing a single byte of what the three built-ins send.
  const legacy = { schemaVersion: 1, id: "prompt-architect", name: "提示词架构师", description: "通用提示词顾问", systemPrompt: "你是提示词架构师。" };
  const composed = composeSystemPrompt(normalizeAssistantPersona(legacy, { id: legacy.id, builtIn: true }));
  assert.equal(composed, `你是提示词架构师。\n\n${PROMPT_PROTOCOL_INSTRUCTION}`);
  assert.equal(composed, composeSystemPrompt({ systemPrompt: "你是提示词架构师。" }), "the old plain-object call must still work");
});

test("the description is picker metadata and never reaches the provider", () => {
  const composed = composeSystemPrompt(normalizeAssistantPersona(character()));
  assert.ok(!composed.includes("偏爱霓虹夜景与胶片颗粒的摄影指导"),
    "the sentence under the name is written for whoever is choosing a character, not for the model");
  assert.ok(!composed.includes("帮我写一段雨后街道的提示词"), "suggested openers are UI affordances, not instructions");
  assert.ok(composed.includes("赛博朋克摄影师"), "the character's name is part of the identity it plays");
});

test("attributes, specialties and rules render into one system message in a fixed order", () => {
  const composed = composeSystemPrompt(normalizeAssistantPersona(character()));
  const order = ["【角色】", "你是一位擅长霓虹夜景的摄影指导。", "【擅长领域】", "【行为准则】", PROMPT_PROTOCOL_INSTRUCTION];
  let cursor = -1;
  for (const marker of order) {
    const at = composed.indexOf(marker);
    assert.ok(at > cursor, `${marker.slice(0, 12)} must follow the previous section`);
    cursor = at;
  }
  assert.match(composed, /语气专业克制[^\n]*；[^\n]*始终用简体中文说明/, "attributes render as one sentence, not a token dump");
  assert.ok(composed.includes("【擅长领域】电影感人像、赛博朋克夜景"));
  assert.ok(composed.includes("1. 先给提示词，再说明改动重点\n2. 不要堆砌无意义的画质词"), "rules are numbered so order is meaningful");
  // The protocol is appended by the control plane and is the last thing the model reads.
  assert.ok(composed.endsWith(PROMPT_PROTOCOL_INSTRUCTION));
});

test("a section renders only when it has content", () => {
  const bare = normalizeAssistantPersona({ id: UUID, name: "沉默", systemPrompt: "只说必要的话。" });
  assert.deepEqual(personaSections(bare), ["只说必要的话。"], "no attributes means no identity block");
  assert.equal(hasPersonaAttributes(bare), false);

  // A name plus one attribute and nothing else is a legitimate character.
  const promptless = normalizeAssistantPersona({ id: UUID, name: "简洁助手", traits: { detail: "concise" } });
  assert.deepEqual(personaSections(promptless), ["【角色】你现在扮演「简洁助手」。\n回复尽量简短，只给必要信息。"]);
  assert.equal(hasPersonaAttributes(promptless), true);

  const unsetTraits = normalizeAssistantPersona({ id: UUID, name: "空", traits: { tone: "", detail: "" }, rules: ["一条规则"] });
  assert.ok(personaSections(unsetTraits)[0].startsWith("【角色】你现在扮演「空」。"));
  assert.ok(!personaSections(unsetTraits)[0].includes("\n"), "with no attribute set the identity block is the name line alone");
});

test("normalisation never throws, repairs junk, and bounds every list", () => {
  for (const junk of [null, undefined, 5, "text", [], { traits: "no" }, { rules: 7 }]) {
    const persona = normalizeAssistantPersona(junk);
    assert.equal(persona.schemaVersion, PERSONA_SCHEMA_VERSION);
    assert.deepEqual(persona.knowledge, []);
    assert.ok(Array.isArray(persona.rules) && Array.isArray(persona.specialties) && Array.isArray(persona.starters));
    for (const entry of PERSONA_ATTRIBUTES) assert.equal(persona.traits[entry.id], "");
  }
  const flooded = normalizeAssistantPersona({
    id: UUID,
    name: "x",
    traits: { tone: "not-a-tone" },
    specialties: Array.from({ length: 40 }, (_unused, index) => `领域 ${index}`),
    rules: Array.from({ length: 40 }, (_unused, index) => `规则 ${index}`),
    starters: Array.from({ length: 40 }, (_unused, index) => `开场 ${index}`),
  });
  assert.equal(flooded.traits.tone, "", "an attribute value outside the vocabulary falls back to unset");
  assert.equal(flooded.specialties.length, MAXIMUM_PERSONA_SPECIALTIES);
  assert.equal(flooded.rules.length, MAXIMUM_PERSONA_RULES);
  assert.equal(flooded.starters.length, MAXIMUM_PERSONA_STARTERS);

  // Lists de-duplicate and drop blanks, so a pasted list with stray lines is usable as typed.
  const pasted = normalizeAssistantPersona({ id: UUID, name: "x", rules: "  一条  \n\n一条\n  另一条  " });
  assert.deepEqual(pasted.rules, ["一条", "另一条"]);
});

test("control characters are stripped from every field, and newlines survive only in the prompt", () => {
  const persona = normalizeAssistantPersona({
    id: UUID,
    name: "断\u0007行",
    description: "\u8bf4\u0007\u660e",
    rules: ["规\u001b则"],
    systemPrompt: "第一段\r\n\r\n第\u0007二段",
  });
  const composed = composeSystemPrompt(persona);
  assert.equal(persona.name, "断行");
  assert.equal(persona.description, "说明");
  assert.deepEqual(persona.rules, ["规则"]);
  // A multi-paragraph character prompt is the normal case, so newlines are kept — but CRLF is
  // normalised, so a file written on Windows composes identically to one written here.
  assert.equal(persona.systemPrompt, "第一段\n\n第二段");
  // Newlines excepted — the composed message is multi-line by design — so this still catches a
  // stray CR surviving normalisation.
  assert.ok(!/[\u0000-\u0009\u000b-\u001f\u007f]/.test(composed), "nothing sent upstream may carry a control character");
});

test("a character that would send nothing is refused, because selecting it would silently do nothing", () => {
  const empty = validateAssistantPersona({ id: UUID, name: "空角色" });
  assert.equal(empty.valid, false);
  assert.deepEqual(empty.errors.map((entry) => entry.code), ["persona_empty"]);

  assert.equal(validateAssistantPersona({ id: UUID, name: "有规则", rules: ["一条"] }).valid, true);
  assert.equal(validateAssistantPersona({ id: UUID, name: "有属性", traits: { tone: "warm" } }).valid, true);
  assert.equal(validateAssistantPersona({ id: UUID, name: "有提示词", systemPrompt: "你是…" }).valid, true);
});

test("a character restating the output protocol is refused, because it breaks the apply button", () => {
  // The control plane appends the real protocol to every system message; a second, differently
  // worded copy is how a model ends up emitting a fence the client cannot recognise.
  const conflicting = validateAssistantPersona({ id: UUID, name: "越权", systemPrompt: "输出时请使用 ```prompt 围栏。" });
  assert.equal(conflicting.valid, false);
  assert.ok(conflicting.errors.some((entry) => entry.code === "persona_protocol_conflict"));
  assert.ok(validateAssistantPersona({ id: UUID, name: "正常", systemPrompt: "输出时请使用英文标签。" }).valid);
});

test("every problem is reported at once so the form can mark each field", () => {
  const broken = validateAssistantPersona({ id: UUID, name: "   ", systemPrompt: "用 ```negative 围栏。" });
  assert.equal(broken.valid, false);
  assert.deepEqual(broken.errors.map((entry) => entry.code).sort(), ["persona_name_missing", "persona_protocol_conflict"]);
  assert.deepEqual([...new Set(broken.errors.map((entry) => entry.field))].sort(), ["name", "systemPrompt"]);
});

test("id shapes say where a character came from", () => {
  // A shipped character is named by its file; one authored in the interface gets a uuid, because
  // that id becomes a filename in the state directory and must not be user-chosen.
  assert.equal(validPersonaId("prompt-architect"), true);
  assert.equal(validUserPersonaId("prompt-architect"), false);
  assert.equal(validUserPersonaId(UUID), true);
  for (const hostile of ["..", "../ui-state", "Prompt-Architect", "", "a".repeat(80), null, 7]) {
    assert.equal(validUserPersonaId(hostile), false, `${hostile} must not be usable as a filename`);
  }
});

test("the library sorts authored characters above the built-ins", () => {
  const sorted = sortAssistantPersonas([
    { id: "a", name: "内置 A", builtIn: true },
    { id: UUID, name: "我的角色", builtIn: false },
    { id: "b", name: "内置 B", builtIn: true },
  ]);
  assert.deepEqual(sorted.map((persona) => persona.name), ["我的角色", "内置 A", "内置 B"]);
});

test("a new character starts usable and uniquely named", () => {
  const draft = draftAssistantPersona();
  assert.equal(hasPersonaAttributes(draft), true, "an empty form would save as an empty character and be refused");
  assert.equal(validateAssistantPersona(draft).valid, true);
  assert.equal(uniquePersonaName("新角色", []), "新角色");
  assert.equal(uniquePersonaName("新角色", ["新角色"]), "新角色 2");
  assert.equal(uniquePersonaName("新角色", ["新角色", "新角色 2"]), "新角色 3");
  assert.equal(uniquePersonaName("  ", []), "新角色");
});
