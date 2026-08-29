import { open } from "node:fs/promises";
import path from "node:path";

const MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024;
export const TRIGGER_REVIEW_SCHEMA = 4;
const IRRELEVANT_PROMPT_TEXT = /(QQ群|群号|qq\s*group|discord|更新|修复|优化|素材|数据集|背景|推荐使用|推荐模型|大模型|插件|adetailer|下载|链接|网址|教程|训练方式|版权|声明|禁止|感谢|建议开启|效果可能|使用方法|权重|步数|采样|分辨率|尺寸|recommend(?:ed)?|checkpoint|dataset|training|download|support|commission|workflow|weight|strength|clip\s*skip|sampler|scheduler|steps?|epochs?|resolution|license|commercial)/i;
const PROSE_PROMPT_TEXT = /\b(reviewed|updated|update|include[ds]?|correct(?:ed)?|fix(?:ed)?|images?|prompts?|please|using|recommended|created|trained|changed|added|removed|works?|etc)\b/i;
const STYLE_SECTION_HEADING = /^(examples?|notes?|details?|description|usage|settings?|recommendations?|showcase|samples?|update log|changelog)$/i;
const CLOTHING_WORDS = /(dress|shirt|skirt|uniform|jacket|coat|kimono|yukata|swimsuit|bikini|underwear|lingerie|pants|shorts|sleeves?|shoes?|footwear|boots?|thighhighs?|stockings?|pantyhose|apron|cardigan|nightwear|pajamas?|hoodie|sailor|capelet|costume|outfit|clothes?|collar|necktie|ribbon|vest|robe|gown|bra|panties|服装|衣服|校服|裙|泳装|睡衣|和服|浴衣|内衣|女仆装)/i;
const HAIR_WORDS = /(hair|bangs|braid|twintails?|ponytail|bun|ahoge|hair ornament|hair bow|hair ribbon|发型|头发|刘海|双马尾|马尾)/i;
const APPEARANCE_WORDS = /(eyes?|skin|face|girl|boy|woman|man|solo|breasts?|chest|body|character|人物|角色|眼睛|肤色|脸|男性|女性)/i;
const POSE_EXPRESSION_WORDS = /(smile|blush|open mouth|closed mouth|looking at viewer|pose|standing|sitting|expression|表情|姿势|微笑)/i;

export function loraMetadataCacheValid(metadata, fileStat, reviewKind = metadata?.triggerReviewKind || "other") {
  return metadata?.detailSchema === 1
    && metadata?.triggerReviewSchema === TRIGGER_REVIEW_SCHEMA
    && metadata?.triggerReviewKind === reviewKind
    && Object.hasOwn(metadata?.promptReview || {}, "versionScopeKind")
    && (!descriptionNeedsVersionIdentity(metadata?.modelDescription || metadata?.description) || typeof metadata?.versionIsLatest === "boolean")
    && Number(metadata.fileSize) === Number(fileStat?.size)
    && Number(metadata.modifiedAt) === Number(fileStat?.mtimeMs)
    && (metadata.previewSchema === 1 || metadata.status !== "found" || Boolean(metadata.previewFile || metadata.previewUrl));
}

function decodeHtmlEntities(value) {
  const named = { amp: "&", apos: "'", gt: ">", lt: "<", nbsp: " ", quot: '"' };
  return value.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (match, entity) => {
    if (entity[0] === "#") {
      const codePoint = Number.parseInt(entity[1].toLowerCase() === "x" ? entity.slice(2) : entity.slice(1), entity[1].toLowerCase() === "x" ? 16 : 10);
      return Number.isFinite(codePoint) && codePoint > 0 && codePoint <= 0x10ffff ? String.fromCodePoint(codePoint) : match;
    }
    return named[entity.toLowerCase()] ?? match;
  });
}

export function plainTextFromHtml(value, maximumLength = 24000) {
  if (typeof value !== "string" || !value.trim()) return "";
  return decodeHtmlEntities(value
    .replace(/<(script|style)\b[^>]*>[\s\S]*?<\/\1>/gi, " ")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/(?:div|h[1-6]|li|ol|p|pre|section|table|tr|ul)>/gi, "\n")
    .replace(/<li\b[^>]*>/gi, "- ")
    .replace(/<[^>]+>/g, " "))
    .replace(/\r/g, "")
    .replace(/[\t ]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
    .slice(0, maximumLength);
}

function parsedJson(value) {
  if (typeof value !== "string") return value;
  const trimmed = value.trim();
  if (!trimmed || !["{", "[", '"'].includes(trimmed[0])) return value;
  try {
    return JSON.parse(trimmed);
  } catch {
    return value;
  }
}

function firstMetadataValue(metadata, keys) {
  for (const key of keys) {
    const value = metadata?.[key];
    if (typeof value === "string" && value.trim()) return value.trim();
    if (typeof value === "number" && Number.isFinite(value)) return String(value);
  }
  return "";
}

function promptWords(value) {
  const source = Array.isArray(value) ? value : [value];
  const words = [];
  for (const item of source) {
    if (typeof item !== "string") continue;
    for (const word of item.split(/[,，;；|\n]+/)) {
      const normalized = word.replace(/^[-*•\s]+/, "").trim();
      if (normalized && normalized.length <= 180 && !words.some((current) => current.toLowerCase() === normalized.toLowerCase())) words.push(normalized);
    }
  }
  return words.slice(0, 80);
}

function canonicalPromptLabel(value) {
  const label = normalizedLabel(value).replace(/[（(].*?[）)]/g, "").trim();
  if (!label || label.length > 32 || /[。！？!?]/.test(label) || IRRELEVANT_PROMPT_TEXT.test(label)) return "";
  if (/^(?:(?:原作|官方)(?:风格|画风)|official (?:art )?style|original (?:art )?style)$/i.test(label)) return "原作画风";
  if (/^(?:人物|角色)(?:基础|默认|触发)(?:词|标|标签|prompt)?$|^(?:character|base)(?:\s+(?:prompt|trigger words?))?$/i.test(label)) return "人物基础词";
  if (/^(?:人物|角色)(?:特征|外貌)(?:词|标|标签|prompt)?$|^character\s+(?:tags?|features?|appearance)$/i.test(label)) return "人物特征";
  if (/^(?:猫娘|猫娘特征|cat girl)$/i.test(label)) return "猫娘特征";
  if (/^(?:两种)?尾巴(?:特征)?$|^(?:2 kinds? of )?tails?$/i.test(label)) return "尾巴";
  if (/^(?:发型|头发|刘海|hair|hairstyle|hair style)$/i.test(label)) return "发型";
  if (/^(?:表情|姿势|动作|expression|pose|expressions?\s*(?:and|&)\s*poses?)$/i.test(label)) return "表情与姿势";
  if (/^(?:配饰|饰品|首饰|accessories?|jewelry)$/i.test(label)) return "配饰";
  if (/^(?:校服|school uniform)$/i.test(label)) return "校服";
  if (/^(?:连衣裙|dress)$/i.test(label)) return "连衣裙";
  if (/^(?:浴衣|yukata)$/i.test(label)) return "浴衣";
  if (/^(?:和服|kimono)$/i.test(label)) return "和服";
  if (/^(?:睡衣|家居服|nightwear|pajamas?)$/i.test(label)) return "睡衣";
  if (/^(?:泳装|泳衣|swimsuit|bikini)$/i.test(label)) return "泳装";
  if (/^(?:私服|日常服|便服|regular (?:outfit|clothes)|casual|casual outfit|daily outfit)$/i.test(label)) return "私服";
  if (/^(?:女仆装|maid|maid outfit)$/i.test(label)) return "女仆装";
  if (/^(?:魔法少女服?|magical girl|magical (?:girl )?(?:uniform|outfit))$/i.test(label)) return "魔法少女服";
  if (/^(?:斗篷兜帽|黑斗篷|兜帽|black cloak|cloak|hood)$/i.test(label)) return "斗篷兜帽";
  if (/^(?:根据需要添加|可选特征|optional features?|use based on you need)$/i.test(label)) return "可选特征";
  if (/^(?:制服|礼服|运动服|内衣|服装|衣服|服饰|穿搭|造型|默认服装|outfits?|clothes?|costumes?)(?:\s*[-#]?\s*\d+)?$/i.test(label)) return /^服装|^衣服|^outfit/i.test(label) && /\d/.test(label) ? label : label.replace(/^outfits?$/i, "服装");
  if (/^(?:风格|画风|style|style trigger|style prompt)$/i.test(label)) return "风格触发词";
  if (/^(?:触发词|关键词|prompt|prompts|trigger|trigger words?|activation text)$/i.test(label)) return "触发词";
  return "";
}

function authorPromptLabel(value) {
  const label = normalizedLabel(value);
  if (!label || label.length > 48 || /[,，;；。！？!?:：\\]/.test(label) || /^https?:/i.test(label)) return "";
  if (IRRELEVANT_PROMPT_TEXT.test(label) || PROSE_PROMPT_TEXT.test(label) || STYLE_SECTION_HEADING.test(label)) return "";
  if (/(?:可以|应该|需要|替换|解锁|加入|推荐|使用|问题|效果|估计|重置|更新|说明|简介|关于|以下|下为|整体|直接|基础训练|about|model|checkpoint|settings?)/i.test(label)) return "";
  return /[\p{L}\p{N}]/u.test(label) ? label : "";
}

function promptTag(value, { allowUnicode = false } = {}) {
  const normalized = value.replace(/^[-*•\s]+/, "").replace(/\s+/g, " ").trim();
  if (!normalized || normalized.length > 96 || /^https?:\/\//i.test(normalized) || IRRELEVANT_PROMPT_TEXT.test(normalized) || PROSE_PROMPT_TEXT.test(normalized)) return "";
  if (/[。！？!?；;]/.test(normalized) || /[\u3400-\u9fff]/.test(normalized)) return "";
  if (!allowUnicode && (!/[a-z0-9]/i.test(normalized) || !/^[a-z0-9_+.'"()\-: ]+$/i.test(normalized))) return "";
  if (allowUnicode && !/[\p{L}\p{N}]/u.test(normalized)) return "";
  if (normalized.split(/\s+/).length > 8) return "";
  return normalized;
}

function reviewedPromptWords(value, { authoritative = false, explicitLabel = false } = {}) {
  if (typeof value !== "string" || !value.trim() || value.length > 1800 || IRRELEVANT_PROMPT_TEXT.test(value)) return [];
  const rawWords = promptWords(value);
  const words = rawWords.map((word) => promptTag(word, { allowUnicode: authoritative || explicitLabel })).filter(Boolean);
  const minimum = authoritative || explicitLabel ? 1 : 3;
  if (words.length < minimum || words.length / Math.max(1, rawWords.length) < (explicitLabel ? .6 : .8)) return [];
  if (!authoritative && !explicitLabel && !/[,，;；|\n]/.test(value)) return [];
  return words;
}

function classifiedPromptLabel(words, fallback = "触发词") {
  const joined = words.join(", ");
  if (/school uniform/i.test(joined)) return "校服";
  if (/nightwear|pajamas?/i.test(joined)) return "睡衣";
  if (/yukata/i.test(joined)) return "浴衣";
  if (/kimono/i.test(joined)) return "和服";
  if (/swimsuit|bikini/i.test(joined)) return "泳装";
  if (/maid outfit|maid dress/i.test(joined)) return "女仆装";
  if (CLOTHING_WORDS.test(joined)) return "服装";
  if (HAIR_WORDS.test(joined) && !APPEARANCE_WORDS.test(joined)) return "发型";
  if (POSE_EXPRESSION_WORDS.test(joined) && !APPEARANCE_WORDS.test(joined)) return "表情与姿势";
  if (APPEARANCE_WORDS.test(joined) || words.length >= 3) return "人物基础词";
  if (/style/i.test(joined)) return "风格触发词";
  return fallback;
}

function parsedVersion(value) {
  const match = String(value || "").match(/(?:^|\s)v(?:ersion)?\s*(\d+)(?:\.(\d+))?/i);
  return match ? { major: Number(match[1]), minor: match[2] === undefined ? null : Number(match[2]) } : null;
}

function parsedVersionPromptHeading(value) {
  const label = normalizedLabel(value);
  const match = label.match(/^(?:v|version)\s*(\d+)(?:\.(\d+|x))?\s+(?:(?:character|style|outfit)\s+)?(?:tags?|prompts?|trigger(?:\s+words?)?|标签|关键词|触发词)$/i);
  if (!match) return null;
  return { major: Number(match[1]), minor: match[2]?.toLowerCase() === "x" ? "x" : match[2] === undefined ? null : Number(match[2]) };
}

const OLDER_PROMPT_BOUNDARY = /(?:下为|以下为|下面是|以下是).{0,12}旧版.{0,12}(?:tag|标签|prompt|关键词)|旧版.{0,12}(?:tag|标签|prompt|关键词)/i;

export function descriptionNeedsVersionIdentity(value) {
  return plainTextFromHtml(value).split("\n").some((line) => OLDER_PROMPT_BOUNDARY.test(line));
}

export function descriptionForLoraVersion(value, versionName, { versionIsLatest } = {}) {
  const description = plainTextFromHtml(value);
  const version = parsedVersion(versionName);
  if (!description || !version) return { description, versionScope: "" };
  const lines = description.split("\n");
  const sections = lines.map((line, index) => ({ index, version: parsedVersionPromptHeading(line) })).filter((item) => item.version);
  if (!sections.length) {
    const olderBoundary = lines.findIndex((line) => OLDER_PROMPT_BOUNDARY.test(line));
    if (olderBoundary < 0) return { description, versionScope: "", versionScopeKind: "" };
    if (typeof versionIsLatest !== "boolean") return { description: "", versionScope: String(versionName || "").trim(), versionScopeKind: "ambiguous" };
    return versionIsLatest
      ? { description: lines.slice(0, olderBoundary).join("\n").trim(), versionScope: String(versionName || "").trim(), versionScopeKind: "latest-before-legacy" }
      : { description: lines.slice(olderBoundary + 1).join("\n").trim(), versionScope: `${String(versionName || "").trim()}（作者旧版区）`, versionScopeKind: "legacy-section" };
  }
  const score = (candidate) => {
    if (candidate.major !== version.major) return -1;
    if (candidate.minor === "x") return 2;
    if (candidate.minor === null) return 1;
    return candidate.minor === version.minor ? 3 : -1;
  };
  // Prefer an exact version section, then a shared V2.x section, then a major-only V3 section.
  const selected = sections.map((section) => ({ ...section, score: score(section.version) })).filter((section) => section.score >= 0).sort((first, second) => second.score - first.score || first.index - second.index)[0];
  if (!selected) return { description: "", versionScope: String(versionName || "").trim() };
  const next = sections.find((section) => section.index > selected.index);
  return {
    description: lines.slice(selected.index + 1, next?.index ?? lines.length).join("\n").trim(),
    versionScope: String(versionName || "").trim(),
    versionScopeKind: "version-heading",
  };
}

function aggregateTagFrequency(value) {
  const totals = new Map();
  const visit = (node) => {
    if (!node || typeof node !== "object") return;
    for (const [key, item] of Object.entries(node)) {
      if (typeof item === "number" && Number.isFinite(item)) totals.set(key, (totals.get(key) || 0) + item);
      else visit(item);
    }
  };
  visit(parsedJson(value));
  return [...totals.entries()]
    .filter(([word]) => word.trim() && word.length <= 180)
    .sort((first, second) => second[1] - first[1])
    .slice(0, 24)
    .map(([word, count]) => ({ word, count }));
}

function safeFieldValue(value) {
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return "";
    if (trimmed.length > 600) return `已解析（${trimmed.length.toLocaleString()} 字符）`;
    return trimmed;
  }
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return "";
}

export function summarizeSafetensorsMetadata(metadata, extension = ".safetensors") {
  const source = metadata && typeof metadata === "object" ? metadata : {};
  const triggerWords = promptWords(parsedJson(firstMetadataValue(source, ["modelspec.trigger_phrase", "modelspec.trigger_phrases", "ss_trigger_words", "trigger_words", "activation text", "activation_text"])));
  const rank = firstMetadataValue(source, ["ss_network_dim", "network_dim", "rank"]);
  const alpha = firstMetadataValue(source, ["ss_network_alpha", "network_alpha", "alpha"]);
  const fields = Object.entries(source)
    .filter(([key]) => !["ss_tag_frequency", "ss_dataset_dirs"].includes(key))
    .map(([key, value]) => ({ key, value: safeFieldValue(value) }))
    .filter((item) => item.value)
    .slice(0, 64);
  return {
    format: extension.replace(/^\./, "").toUpperCase() || "UNKNOWN",
    parsed: Object.keys(source).length > 0,
    fieldCount: Object.keys(source).length,
    title: firstMetadataValue(source, ["modelspec.title", "ss_output_name", "title"]),
    author: firstMetadataValue(source, ["modelspec.author", "author"]),
    description: plainTextFromHtml(firstMetadataValue(source, ["modelspec.description", "description"]), 8000),
    baseModel: firstMetadataValue(source, ["modelspec.base_model", "ss_sd_model_name", "ss_base_model_version", "base_model"]),
    architecture: firstMetadataValue(source, ["modelspec.architecture", "ss_base_model_version", "architecture"]),
    implementation: firstMetadataValue(source, ["modelspec.implementation", "ss_network_module", "network_module"]),
    resolution: firstMetadataValue(source, ["modelspec.resolution", "ss_resolution", "resolution"]),
    rank,
    alpha,
    network: [rank && `Rank ${rank}`, alpha && `Alpha ${alpha}`].filter(Boolean).join(" / "),
    epochs: firstMetadataValue(source, ["ss_num_epochs", "num_epochs", "epochs"]),
    steps: firstMetadataValue(source, ["ss_steps", "ss_max_train_steps", "max_train_steps", "steps"]),
    trainingImages: firstMetadataValue(source, ["ss_num_train_images", "num_train_images", "training_images"]),
    optimizer: firstMetadataValue(source, ["ss_optimizer", "optimizer"]),
    date: firstMetadataValue(source, ["modelspec.date", "ss_training_finished_at", "date"]),
    triggerWords,
    topTags: aggregateTagFrequency(source.ss_tag_frequency),
    fields,
  };
}

export async function readLoraFileMetadata(modelPath) {
  const extension = path.extname(modelPath).toLowerCase();
  if (extension !== ".safetensors") {
    return summarizeSafetensorsMetadata({}, extension);
  }
  const handle = await open(modelPath, "r");
  try {
    const fileStat = await handle.stat();
    if (fileStat.size < 10) throw new Error("Safetensors file is too small");
    const lengthBuffer = Buffer.alloc(8);
    const lengthRead = await handle.read(lengthBuffer, 0, 8, 0);
    if (lengthRead.bytesRead !== 8) throw new Error("Safetensors header length is incomplete");
    const headerLength = Number(lengthBuffer.readBigUInt64LE(0));
    if (!Number.isSafeInteger(headerLength) || headerLength <= 1 || headerLength > MAX_SAFETENSORS_HEADER_BYTES || headerLength + 8 > fileStat.size) {
      throw new Error("Safetensors header length is invalid");
    }
    const headerBuffer = Buffer.alloc(headerLength);
    const headerRead = await handle.read(headerBuffer, 0, headerLength, 8);
    if (headerRead.bytesRead !== headerLength) throw new Error("Safetensors header is incomplete");
    const header = JSON.parse(headerBuffer.toString("utf8"));
    return summarizeSafetensorsMetadata(header.__metadata__, extension);
  } finally {
    await handle.close();
  }
}

function normalizedLabel(value) {
  return value.replace(/^\s*(?:[-*•]|\d+[.、])\s*/, "").replace(/[：:]$/, "").trim().slice(0, 48);
}

function reviewStyleTriggers({ trainedWords, description, modelDescription, versionDescription, localMetadata, versionScope }) {
  const words = [];
  const wordSet = new Set();
  const reviewedSources = [];
  const sourceSet = new Set();
  let ignoredSegments = 0;
  const addSource = (id, label) => {
    if (sourceSet.has(id)) return;
    sourceSet.add(id);
    reviewedSources.push({ id, label });
  };
  const addWords = (values) => {
    let added = 0;
    for (const word of values) {
      const normalized = promptTag(word, { allowUnicode: true });
      const key = normalized.toLowerCase();
      if (!normalized || wordSet.has(key)) continue;
      wordSet.add(key);
      words.push(normalized);
      added += 1;
    }
    return added;
  };
  const reviewDescription = (value, source, sourceLabel) => {
    const lines = plainTextFromHtml(value).split("\n").map((line) => line.replace(/^[-*•]\s*/, "").trim()).filter(Boolean);
    if (!lines.length) return;
    addSource(source, sourceLabel);
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index];
      const inline = line.match(/^(.{1,48}?)[：:]\s*(.+)$/);
      const label = inline ? canonicalPromptLabel(inline[1]) : canonicalPromptLabel(line);
      if (!["触发词", "风格触发词"].includes(label)) {
        if (inline || label) {
          ignoredSegments += 1;
          continue;
        }
        const unlabelled = STYLE_SECTION_HEADING.test(line) ? [] : reviewedPromptWords(line, { authoritative: true });
        if (unlabelled.length && addWords(unlabelled)) continue;
        ignoredSegments += 1;
        continue;
      }
      const promptText = inline ? inline[2] : lines[index + 1] || "";
      const accepted = reviewedPromptWords(promptText, { explicitLabel: true });
      if (addWords(accepted)) {
        if (!inline) index += 1;
      } else {
        ignoredSegments += 1;
      }
    }
  };

  reviewDescription(versionDescription, "civitai-detail", "Civitai Detail");
  reviewDescription(modelDescription, "civitai-description", "Civitai 简介");
  if (!versionDescription && !modelDescription) reviewDescription(description, "civitai-description", "Civitai 简介");
  if (Array.isArray(trainedWords) && trainedWords.some((item) => typeof item === "string" && item.trim())) {
    addSource("civitai-trained-words", "Civitai 触发词");
    for (const item of trainedWords) {
      const accepted = reviewedPromptWords(item, { authoritative: true });
      if (!addWords(accepted)) ignoredSegments += 1;
    }
  }
  const localWords = reviewedPromptWords((localMetadata?.triggerWords || []).join(", "), { authoritative: true });
  if (localWords.length) {
    addSource("safetensors", "本地模型元数据");
    if (!addWords(localWords)) ignoredSegments += 1;
  }
  const groups = words.length ? [{ label: "触发词", words: words.slice(0, 40), source: "reviewed-style-triggers", sources: reviewedSources.map((source) => source.id) }] : [];
  return { groups, reviewedSources, ignoredSegments, acceptedGroups: groups.length, reviewKind: "style", versionScope: versionScope.value, versionScopeKind: versionScope.kind };
}

export function reviewLoraPrompts({ trainedWords = [], description = "", modelDescription = "", versionDescription = "", versionName = "", versionIsLatest, localMetadata = null, reviewKind = "other" } = {}) {
  const versionOptions = { versionIsLatest };
  const scopedVersion = descriptionForLoraVersion(versionDescription, versionName, versionOptions);
  const scopedModel = descriptionForLoraVersion(modelDescription, versionName, versionOptions);
  const scopedFallback = descriptionForLoraVersion(description, versionName, versionOptions);
  const duplicateDescriptions = Boolean(versionDescription && modelDescription && plainTextFromHtml(versionDescription) === plainTextFromHtml(modelDescription));
  const reviewedVersionDescription = duplicateDescriptions ? "" : scopedVersion.description;
  const reviewedModelDescription = scopedModel.description;
  const reviewedDescription = scopedFallback.description;
  const scopedSource = scopedVersion.versionScope ? scopedVersion : scopedModel.versionScope ? scopedModel : (!modelDescription && !versionDescription ? scopedFallback : {});
  const versionScope = { value: scopedSource.versionScope || "", kind: scopedSource.versionScopeKind || "" };
  if (reviewKind === "style") return reviewStyleTriggers({ trainedWords, description: reviewedDescription, modelDescription: reviewedModelDescription, versionDescription: reviewedVersionDescription, localMetadata, versionScope });
  const groups = [];
  const signatures = new Set();
  const coveredWords = new Set();
  const reviewedSources = [];
  let ignoredSegments = 0;
  const addGroup = (label, words, source) => {
    const normalizedWords = words.map((word) => promptTag(word, { allowUnicode: true })).filter(Boolean).slice(0, 80);
    const signature = normalizedWords.map((word) => word.toLowerCase()).join("\u0000");
    if (!label || !normalizedWords.length || signatures.has(signature) || groups.length >= 32) return false;
    signatures.add(signature);
    normalizedWords.forEach((word) => coveredWords.add(word.toLowerCase()));
    groups.push({ label, words: normalizedWords, source });
    return true;
  };

  const reviewDescription = (value, source, sourceLabel) => {
    const lines = plainTextFromHtml(value).split("\n").map((line) => line.replace(/^[-*•]\s*/, "").trim()).filter(Boolean);
    if (!lines.length) return;
    reviewedSources.push({ id: source, label: sourceLabel });
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index];
      const inline = line.match(/^(.{1,48}?)[：:]\s*(.+)$/);
      const inlineLabel = inline ? authorPromptLabel(inline[1]) : "";
      if (inlineLabel) {
        const words = reviewedPromptWords(inline[2], { explicitLabel: true });
        if (!addGroup(inlineLabel, words, source)) ignoredSegments += 1;
        continue;
      }
      const heading = authorPromptLabel(line);
      if (heading) {
        let accepted = false;
        while (index + 1 < lines.length && !/^(.{1,48}?)[：:]/.test(lines[index + 1])) {
          const nextLine = lines[index + 1];
          const nextHeading = authorPromptLabel(nextLine);
          const followingLine = lines[index + 2] || "";
          const followingWords = reviewedPromptWords(followingLine, { explicitLabel: true });
          const followingIsPrompt = /[,，;；|]/.test(followingLine)
            ? followingWords.length > 0
            : followingWords.length === 1 && /^[a-z0-9_+.'()-]+$/i.test(followingWords[0]);
          if (nextHeading && followingIsPrompt) break;
          const words = reviewedPromptWords(nextLine, { explicitLabel: true });
          if (!words.length) break;
          accepted = addGroup(heading, words, source) || accepted;
          index += 1;
        }
        if (!accepted) ignoredSegments += 1;
        continue;
      }
      const alternate = line.match(/^(.{1,24}?)还可以把\s*([^，,。]+?)\s*替换为\s*(.+?)(?:来|以便)?解锁(.+?)(?:形态|外观)/i);
      if (alternate) {
        const baseLabel = authorPromptLabel(alternate[1]);
        const baseGroup = [...groups].reverse().find((group) => group.label === baseLabel);
        const removed = promptTag(alternate[2], { allowUnicode: true });
        const replacements = reviewedPromptWords(alternate[3], { explicitLabel: true });
        const variantName = normalizedLabel(alternate[4]) || "另一种";
        if (baseGroup && removed && replacements.length) {
          const words = baseGroup.words.flatMap((word) => word.toLowerCase() === removed.toLowerCase() ? replacements : [word]);
          if (addGroup(`${baseLabel}（${variantName}形态）`, words, source)) continue;
        }
      }
      const words = reviewedPromptWords(line);
      if (words.length) addGroup(classifiedPromptLabel(words), words, source);
      else ignoredSegments += 1;
    }
  };

  reviewDescription(reviewedVersionDescription, "civitai-detail", "Civitai Detail");
  reviewDescription(reviewedModelDescription, "civitai-description", versionScope.value ? `Civitai 简介（${versionScope.value}）` : "Civitai 简介");
  if (!reviewedVersionDescription && !reviewedModelDescription) reviewDescription(reviewedDescription, "civitai-description", versionScope.value ? `Civitai 简介（${versionScope.value}）` : "Civitai 简介");

  if (Array.isArray(trainedWords) && trainedWords.some((item) => typeof item === "string" && item.trim())) {
    reviewedSources.push({ id: "civitai-trained-words", label: "Civitai 触发词" });
    for (const item of trainedWords) {
      const words = reviewedPromptWords(item, { authoritative: true });
      if (words.length && words.every((word) => coveredWords.has(word.toLowerCase()))) {
        ignoredSegments += 1;
        continue;
      }
      if (!addGroup(classifiedPromptLabel(words), words, "civitai-trained-words")) ignoredSegments += 1;
    }
  }

  const localWords = reviewedPromptWords((localMetadata?.triggerWords || []).join(", "), { authoritative: true });
  if (localWords.length) {
    reviewedSources.push({ id: "safetensors", label: "本地模型元数据" });
    if (!localWords.every((word) => coveredWords.has(word.toLowerCase()))) addGroup(classifiedPromptLabel(localWords), localWords, "safetensors");
    else ignoredSegments += 1;
  }

  const labelTotals = new Map();
  for (const group of groups) labelTotals.set(group.label, (labelTotals.get(group.label) || 0) + 1);
  const labelIndexes = new Map();
  const reviewedGroups = groups.map((group) => {
    if (labelTotals.get(group.label) < 2) return group;
    const index = (labelIndexes.get(group.label) || 0) + 1;
    labelIndexes.set(group.label, index);
    return { ...group, label: `${group.label} ${index}` };
  });
  return { groups: reviewedGroups, reviewedSources, ignoredSegments, acceptedGroups: reviewedGroups.length, reviewKind, versionScope: versionScope.value, versionScopeKind: versionScope.kind };
}

export function buildTriggerGroups(input = {}) {
  return reviewLoraPrompts(input).groups;
}
