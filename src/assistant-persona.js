// The AI character record: what a persona *is*, independent of where it is stored.
//
// A character used to be a hand-written JSON file holding one prose `systemPrompt`. That is still
// a valid character — the three shipped ones are exactly that — but the character interface needs
// something a form can edit, so the record now also carries named attributes (tone, verbosity,
// reply language, what to do when the brief is thin), a list of specialties, a list of behaviour
// rules, and suggested openers.
//
// The structured fields are not a second prompt: `personaSections` renders them into the same
// system message the prose goes into, in a fixed order, and the editor previews the result through
// this very function. What you see in the preview is what the provider receives.
//
// Deliberately pure and dependency-free so the browser, the control plane and the tests all agree
// on one definition of a character.

export const PERSONA_SCHEMA_VERSION = 2;

export const MAXIMUM_PERSONA_NAME = 80;
export const MAXIMUM_PERSONA_DESCRIPTION = 240;
export const MAXIMUM_SYSTEM_PROMPT_CHARACTERS = 20000;
export const MAXIMUM_PERSONA_RULES = 12;
export const MAXIMUM_PERSONA_RULE = 200;
export const MAXIMUM_PERSONA_SPECIALTIES = 12;
export const MAXIMUM_PERSONA_SPECIALTY = 40;
export const MAXIMUM_PERSONA_STARTERS = 6;
export const MAXIMUM_PERSONA_STARTER = 80;

// Ids come from two places and the shape says which. A shipped character is named by its file
// (`prompt-architect`); one authored in the interface gets a server-generated uuid, because that id
// becomes a filename in the state directory and must not be user-chosen.
export const BUILT_IN_PERSONA_ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
export const USER_PERSONA_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const PERSONA_ID_MAXIMUM_LENGTH = 64;

export function validUserPersonaId(value) {
  return typeof value === "string" && USER_PERSONA_ID_PATTERN.test(value);
}

export function validPersonaId(value) {
  if (typeof value !== "string" || !value || value.length > PERSONA_ID_MAXIMUM_LENGTH) return false;
  return BUILT_IN_PERSONA_ID_PATTERN.test(value) || USER_PERSONA_ID_PATTERN.test(value);
}

// Attribute vocabulary. Values are ASCII tokens because they are stored and compared; labels are
// Chinese because they are read. Unlike a model id or an effort tier these never reach a provider
// verbatim — `personaSections` renders the *sentence* below — so there is no reason to show the
// token to the user.
const attribute = (id, label, hint, choices) => Object.freeze({
  id,
  label,
  hint,
  choices: Object.freeze(choices.map(([value, choiceLabel, sentence]) => Object.freeze({ value, label: choiceLabel, sentence }))),
});

export const PERSONA_ATTRIBUTES = Object.freeze([
  attribute("tone", "语气", "决定回复读起来的性格。", [
    ["", "不指定", ""],
    ["warm", "亲切随和", "语气亲切随和，像一位耐心的伙伴"],
    ["professional", "专业克制", "语气专业克制，不说客套话"],
    ["playful", "活泼跳脱", "语气活泼，可以适度玩梗，但不喧宾夺主"],
    ["blunt", "直接简短", "语气直接，先给结论再补充理由"],
  ]),
  attribute("detail", "详略", "决定一次回复给多少内容。", [
    ["", "不指定", ""],
    ["concise", "简洁", "回复尽量简短，只给必要信息"],
    ["balanced", "均衡", "回复长度适中，关键处展开、其余从简"],
    ["thorough", "详尽", "回复可以展开讲解，给出取舍理由与备选方案"],
  ]),
  attribute("language", "回复语言", "只影响说明文字；提示词本身始终是英文标签。", [
    ["", "不指定", ""],
    ["zh", "简体中文", "始终用简体中文说明"],
    ["en", "English", "始终用英文说明"],
    ["follow", "跟随用户", "用用户当前使用的语言说明"],
  ]),
  attribute("initiative", "信息不足时", "决定它先问还是先做。", [
    ["", "不指定", ""],
    ["ask-first", "先提问", "用户描述不清时，先用一两个问题问清关键信息，再产出"],
    ["assume", "先产出", "用户描述不清时，按最合理的假设直接产出，并说明你假设了什么"],
  ]),
]);

const ATTRIBUTE_BY_ID = new Map(PERSONA_ATTRIBUTES.map((entry) => [entry.id, entry]));

export function personaAttributeSentence(id, value) {
  const found = ATTRIBUTE_BY_ID.get(id)?.choices.find((choice) => choice.value === value);
  return found?.sentence || "";
}

// Stripped everywhere: these fields are pasted into by users and then rendered into a system
// message, so a stray control character would travel upstream inside the prompt.
const CONTROL_CHARACTERS = /[\u0000-\u001f\u007f]/g;
// The block form keeps newlines — a multi-paragraph character prompt is the normal case — and
// normalises CRLF first so a file written on Windows composes identically to one written here.
const CONTROL_CHARACTERS_KEEPING_NEWLINES = /[\u0000-\u0009\u000b-\u001f\u007f]/g;

function cleanLine(value, limit) {
  const raw = typeof value === "string" ? value : "";
  return raw.replace(CONTROL_CHARACTERS, "").replace(/\s+/g, " ").trim().slice(0, limit);
}

function cleanBlock(value, limit) {
  const raw = typeof value === "string" ? value : "";
  return raw.replace(/\r\n?/g, "\n").replace(CONTROL_CHARACTERS_KEEPING_NEWLINES, "").trim().slice(0, limit);
}

function cleanList(value, { maximum, itemLimit }) {
  const rows = Array.isArray(value)
    ? value
    : typeof value === "string" ? value.split("\n") : [];
  const seen = new Set();
  const items = [];
  for (const row of rows) {
    const line = cleanLine(row, itemLimit);
    if (!line || seen.has(line)) continue;
    seen.add(line);
    items.push(line);
    if (items.length >= maximum) break;
  }
  return items;
}

function normalizeTraits(value) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const traits = {};
  for (const entry of PERSONA_ATTRIBUTES) {
    const candidate = typeof source[entry.id] === "string" ? source[entry.id] : "";
    traits[entry.id] = entry.choices.some((choice) => choice.value === candidate) ? candidate : "";
  }
  return traits;
}

export function hasPersonaAttributes(persona) {
  return PERSONA_ATTRIBUTES.some((entry) => persona?.traits?.[entry.id])
    || Boolean(persona?.specialties?.length)
    || Boolean(persona?.rules?.length);
}

// Never throws and never returns null: the character interface has to be able to open a damaged
// record in order to repair it. Whether a record is *usable* is a separate question, answered by
// `validateAssistantPersona`.
export function normalizeAssistantPersona(value, { id, builtIn = false } = {}) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const stated = typeof source.id === "string" ? source.id.trim().toLowerCase() : "";
  const resolvedId = validPersonaId(stated) ? stated : (validPersonaId(id) ? id : "");
  return {
    schemaVersion: PERSONA_SCHEMA_VERSION,
    id: resolvedId,
    builtIn: Boolean(builtIn),
    name: cleanLine(source.name, MAXIMUM_PERSONA_NAME) || resolvedId,
    description: cleanLine(source.description, MAXIMUM_PERSONA_DESCRIPTION),
    traits: normalizeTraits(source.traits),
    specialties: cleanList(source.specialties, { maximum: MAXIMUM_PERSONA_SPECIALTIES, itemLimit: MAXIMUM_PERSONA_SPECIALTY }),
    rules: cleanList(source.rules, { maximum: MAXIMUM_PERSONA_RULES, itemLimit: MAXIMUM_PERSONA_RULE }),
    systemPrompt: cleanBlock(source.systemPrompt, MAXIMUM_SYSTEM_PROMPT_CHARACTERS),
    starters: cleanList(source.starters, { maximum: MAXIMUM_PERSONA_STARTERS, itemLimit: MAXIMUM_PERSONA_STARTER }),
    // Retrieval is not implemented; the field is reported empty regardless of what a file says, so
    // no build can start shipping knowledge payloads before the retrieval stage exists.
    knowledge: [],
  };
}

// The parts of a character that reach the provider, in the order they are sent.
//
// Two rules hold this together. A section renders only when it has content, so a prose-only
// character — every v1 file, including the three shipped ones — composes byte-for-byte as it did
// before attributes existed. And `description` is never rendered: it is the sentence under the
// name in the picker, written for whoever is choosing a character, not for the model.
export function personaSections(persona) {
  if (!persona) return [];
  const sections = [];
  const name = cleanLine(persona.name, MAXIMUM_PERSONA_NAME);
  const sentences = PERSONA_ATTRIBUTES
    .map((entry) => personaAttributeSentence(entry.id, persona.traits?.[entry.id]))
    .filter(Boolean);
  // The identity block exists to state attributes, so a character that declares none does not get
  // one. Its prose already says who it is.
  if (name && hasPersonaAttributes(persona)) {
    const lines = [`【角色】你现在扮演「${name}」。`];
    if (sentences.length) lines.push(`${sentences.join("；")}。`);
    sections.push(lines.join("\n"));
  }
  const base = cleanBlock(persona.systemPrompt, MAXIMUM_SYSTEM_PROMPT_CHARACTERS);
  if (base) sections.push(base);
  const specialties = Array.isArray(persona.specialties) ? persona.specialties.filter(Boolean) : [];
  if (specialties.length) sections.push(`【擅长领域】${specialties.join("、")}`);
  const rules = Array.isArray(persona.rules) ? persona.rules.filter(Boolean) : [];
  if (rules.length) {
    sections.push(["【行为准则】", ...rules.map((rule, index) => `${index + 1}. ${rule}`)].join("\n"));
  }
  return sections;
}

function invalid(field, code, message) {
  return { field, code, message };
}

// A character that restates the output protocol would break the apply button: the control plane
// appends the real one to every system message, and a second, differently-worded copy is how a
// model ends up emitting a fence the client cannot recognise.
const PROTOCOL_CONFLICT_PATTERN = /```\s*(?:prompt|negative)/i;

export const PERSONA_ERROR_CODES = Object.freeze([
  "persona_name_missing",
  "persona_empty",
  "persona_protocol_conflict",
  "persona_prompt_too_long",
  "persona_readonly",
  "persona_limit_reached",
]);

// Reports every problem at once, like the settings validator, so the form can mark each offending
// field instead of making the user fix one per round trip. Inspects the *raw* input for the same
// reason it does there: normalisation truncates and drops, and validating afterwards would report
// success for a value the user never wrote.
export function validateAssistantPersona(value) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const persona = normalizeAssistantPersona(source, { id: typeof source.id === "string" ? source.id : "" });
  const errors = [];
  if (!cleanLine(source.name, MAXIMUM_PERSONA_NAME)) {
    errors.push(invalid("name", "persona_name_missing", "请填写角色名称。"));
  }
  const rawPrompt = typeof source.systemPrompt === "string" ? source.systemPrompt : "";
  if (rawPrompt.length > MAXIMUM_SYSTEM_PROMPT_CHARACTERS) {
    errors.push(invalid("systemPrompt", "persona_prompt_too_long", `角色提示词超过 ${MAXIMUM_SYSTEM_PROMPT_CHARACTERS} 个字符。`));
  }
  if (PROTOCOL_CONFLICT_PATTERN.test(rawPrompt)) {
    errors.push(invalid("systemPrompt", "persona_protocol_conflict", "角色提示词中不要再写 ```prompt / ```negative 输出协议，控制面会自动追加；重复声明会导致「应用到提示词」失效。"));
  }
  // Nothing but the appended protocol would be sent, which is the same as having no character at
  // all — and it would silently look like the selection had no effect.
  if (!personaSections(persona).length) {
    errors.push(invalid("systemPrompt", "persona_empty", "这个角色还没有任何内容：请填写角色提示词，或至少设置一项属性、擅长领域或行为准则。"));
  }
  return { valid: errors.length === 0, errors, persona };
}

// Ordering for the picker and the character list: the characters a user wrote come first, because
// those are the ones they came here to manage, and the built-ins sit below as a stable tail.
export function sortAssistantPersonas(personas) {
  return [...(Array.isArray(personas) ? personas : [])].sort((first, second) => {
    if (Boolean(first.builtIn) !== Boolean(second.builtIn)) return first.builtIn ? 1 : -1;
    return String(first.name).localeCompare(String(second.name), "zh-Hans-CN");
  });
}

// Suffixes rather than refusing, exactly as a duplicated service configuration does: a colliding
// name is a naming inconvenience, not a data problem.
export function uniquePersonaName(name, taken) {
  const existing = new Set((taken || []).map((entry) => String(entry).trim()));
  const base = cleanLine(name, MAXIMUM_PERSONA_NAME) || "新角色";
  if (!existing.has(base)) return base;
  for (let index = 2; index <= 99; index += 1) {
    const suffix = ` ${index}`;
    const candidate = `${base.slice(0, MAXIMUM_PERSONA_NAME - suffix.length)}${suffix}`;
    if (!existing.has(candidate)) return candidate;
  }
  return base;
}

// The starting point for 新建角色. Empty enough that the form is obviously the user's to fill in,
// complete enough that saving it immediately produces a character that actually does something.
export function draftAssistantPersona() {
  return normalizeAssistantPersona({
    name: "新角色",
    description: "",
    traits: { tone: "professional", detail: "balanced", language: "zh", initiative: "ask-first" },
    specialties: [],
    rules: [],
    systemPrompt: "",
    starters: [],
  });
}
