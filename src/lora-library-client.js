// Published version transport. Broadcasts carry ONLY a notification/revision;
// bindings always come back from the authoritative server, never another tab.
import { admitUiStateResponse } from "./ui-state-authority.js";
export const LORA_PUBLICATION_EVENT = "xirai-lora-publication";
const resources = new Map();
let revision = 0, sequence = 0, channel, polling = false;
const clone = (value) => value === undefined ? undefined : structuredClone(value);
// Rebase against what this UI last acknowledged, not a merged PUT response it
// never rendered. Otherwise a quiet tab's unchanged old Prompt looks like a new
// local edit after another tab's later edit, and the next publication loses it.
export function publishedUiSnapshot() { const basis = resources.get("/api/ui-state"); return clone(basis?.localSnapshot ?? basis?.snapshot); }
export function publishedCardSnapshot() { const basis = resources.get("/api/lora-cards"); return clone(basis?.localSnapshot ?? basis?.snapshot); }
export function currentLoraRevision() { return revision; }
export function announceLoraPublication(next, { broadcast = true } = {}) {
  if (!Number.isSafeInteger(next) || next <= revision) return;
  revision = next;
  if (typeof window !== "undefined") window.dispatchEvent(new CustomEvent(LORA_PUBLICATION_EVENT, { detail: { revision: next } }));
  if (broadcast) channel?.postMessage({ type: "published", revision: next });
}
export function startLoraPublicationWatch() {
  if (typeof window === "undefined" || polling) return;
  polling = true;
  if (typeof BroadcastChannel !== "undefined") {
    channel = new BroadcastChannel("xirai-lora-published-v1");
    channel.onmessage = ({ data }) => { if (data?.type === "published") announceLoraPublication(data.revision, { broadcast: false }); };
  }
  const poll = async () => {
    try {
      const response = await globalThis.fetch("/api/lora-relink/state", { cache: "no-store" });
      if (response.ok) { const value = await response.json(); if (value.revision > revision) announceLoraPublication(value.revision); }
    } catch { /* The ordinary workspace remains usable if repair is unavailable. */ }
    window.setTimeout(poll, 2500);
  };
  window.setTimeout(poll, 2500);
}
export async function resolveLoraReferences(engine, references, { history = false, signal } = {}) {
  const response = await globalThis.fetch("/api/lora-relink/resolve", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ engine, references, history }), signal });
  const payload = await response.json();
  if (!response.ok) throw Object.assign(new Error(payload.error || "LoRA 引用不可解析，请重新选择目标。"), { status: response.status, ...(payload.resolution ? { resolution: payload.resolution } : {}) });
  return payload;
}
export async function loraFetch(input, options = {}) {
  const url = typeof input === "string" ? input : input?.url || "";
  const pathname = url.startsWith("/") ? url.split("?")[0] : (() => { try { return new URL(url).pathname; } catch { return ""; } })();
  const method = String(options.method || "GET").toUpperCase();
  const owned = pathname === "/api/ui-state" || pathname === "/api/lora-cards";
  const { loraReferences, loraObserveOnly = false, ...nativeOptions } = options;
  if (loraObserveOnly && method !== "GET") throw new Error("只读校验不能用于保存。");
  const headers = new Headers(options.headers || {});
  const basis = resources.get(pathname);
  if ((owned && method !== "GET") || (pathname === "/api/lora-cards/asset" && method === "POST")) {
    headers.set("X-XirAI-LoRA-Revision", String(basis?.revision ?? revision));
    if (basis?.version) headers.set("X-XirAI-Document-Version", basis.version);
    if (pathname === "/api/ui-state") headers.set("Prefer", "return=representation");
  }
  let body = options.body;
  if (pathname === "/api/inference/jobs" && method === "POST" && typeof body === "string") {
    const request = JSON.parse(body);
    const refs = loraReferences || request.gallery_settings?.loras || (request.loras || []).map((item) => ({ value: item.path, weight: item.weight }));
    if (refs.length) {
      const resolved = await resolveLoraReferences(request.engine, refs, { signal: options.signal });
      request.loras = resolved.loras;
      headers.set("X-XirAI-LoRA-Revision", String(resolved.revision));
      headers.set("X-XirAI-LoRA-Context", JSON.stringify(resolved.references.map(({ instanceId, contentHash }) => ({ instanceId, contentHash }))));
      body = JSON.stringify(request);
    } else headers.set("X-XirAI-LoRA-Revision", String(revision));
  }
  const serial = ++sequence;
  const response = await globalThis.fetch(input, { ...nativeOptions, headers, body });
  const observed = Number(response.headers?.get("X-XirAI-LoRA-Revision"));
  const hasVersion = response.headers?.has("X-XirAI-LoRA-Revision");
  if (hasVersion && observed < revision) throw Object.assign(new Error("迟到的修复前响应已丢弃，请重新读取。"), { status: 409 });
  if (owned && response.ok) {
    const current = resources.get(pathname);
    if (current && serial < current.serial) throw Object.assign(new Error("迟到的工作区响应已丢弃。"), { status: 409 });
    // A successful status/header is not a read receipt. Never publish a basis
    // from incomplete JSON or a UI document the hydration owner cannot admit.
    const payload = response.status === 204 && pathname === "/api/lora-cards"
      ? undefined : await response.clone().json();
    if (pathname === "/api/ui-state") {
      const admitted = admitUiStateResponse(payload);
      if (method !== "GET" && admitted.status === "confirmed-absent") throw new Error("保存确认缺少页面状态；请保留本页编辑。");
    }
    if (options.signal?.aborted) throw options.signal.reason || new Error("已取消的工作区响应已丢弃。");
    if (resources.get(pathname)?.serial > serial) throw Object.assign(new Error("迟到的工作区响应已丢弃。"), { status: 409 });
    const snapshot = pathname === "/api/ui-state" ? payload?.state : payload?.store;
    let localSnapshot = snapshot;
    if (method === "PUT" && typeof body === "string") {
      try { const submitted = JSON.parse(body); localSnapshot = pathname === "/api/ui-state" ? submitted : submitted.store; } catch { /* No unparsed body is a rebase basis. */ }
    }
    if (!loraObserveOnly) resources.set(pathname, { serial, revision: hasVersion ? observed : 0,
      version: response.headers?.get("X-XirAI-Document-Version"), snapshot: clone(snapshot), localSnapshot: clone(localSnapshot) });
  }
  if (!loraObserveOnly && hasVersion && observed > revision && response.ok) announceLoraPublication(observed);
  return response;
}

const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
/** Rebase only name spans actually changed by the authoritative publication.
 * Equal non-name bytes prove this is a confirmed tag-only patch, not a prompt
 * replacement. Derive a per-field exact qualified-name mapping only when ALL
 * baseline occurrences agree with the publication. Match parsed local spans by
 * that evidence, not offsets/counts: insertion/deletion/reordering/duplicates
 * must not revive a confirmed old name on the next fresh-version save. Bare or
 * user-changed/unconfirmed names, weights and non-tag text remain untouched.
 */
export function rebasePublishedTagSpans(before, local, fresh) {
  if (![before, local, fresh].every((value) => typeof value === "string")) return local;
  const parse = (text) => {
    const tags = [...text.matchAll(/<lora:([^<>:]+)(?::[^<>]*)?>/g)].map((match) => ({
      name: match[1], start: match.index + 6, end: match.index + 6 + match[1].length,
    }));
    let skeleton = text;
    for (const tag of [...tags].reverse()) skeleton = skeleton.slice(0, tag.start) + "\u0000" + skeleton.slice(tag.end);
    return { tags, skeleton };
  };
  const old = parse(before), next = parse(fresh), current = parse(local);
  if (!old.tags.length || old.skeleton !== next.skeleton) return local;
  const names = new Map();
  for (let index = 0; index < old.tags.length; index += 1) {
    const from = old.tags[index].name, to = next.tags[index].name;
    if (!from.includes("/")) continue;
    // An unchanged or differently patched duplicate is not unanimous consent.
    if (!names.has(from)) names.set(from, to);
    else if (names.get(from) !== to) names.set(from, null);
  }
  let result = local;
  for (const live of [...current.tags].reverse()) {
    const target = names.get(live.name);
    if (!target || target === live.name) continue;
    result = result.slice(0, live.start) + target + result.slice(live.end);
  }
  return result;
}
/** Preserve later local tuning/text while adopting authoritative associations.
 * No union/dedup of lists, no adding weights together. Server membership/order
 * wins; local member tuning follows its explicitly audited source association.
 */
export function rebasePublishedWorkspace(base = {}, local = {}, fresh = {}) {
  base ||= {}; local ||= {}; fresh ||= {};
  const merge = (a, b, c) => {
    if (equal(a, b)) return clone(c);
    if (typeof a === "string" && typeof b === "string" && typeof c === "string") return rebasePublishedTagSpans(a, b, c);
    // Preset/group IDs, not array offsets, align concurrently edited text.
    // Preserve local membership/order; mounted bindings are handled below.
    if ([a, b, c].every(Array.isArray)) return b.map((item) => {
      const old = item?.id && a.find((entry) => entry?.id === item.id), fresh = item?.id && c.find((entry) => entry?.id === item.id);
      return old && fresh ? merge(old, item, fresh) : clone(item);
    });
    if (!a || !b || !c || typeof a !== "object" || typeof b !== "object" || typeof c !== "object" || Array.isArray(a) || Array.isArray(b) || Array.isArray(c)) return clone(b);
    const result = {};
    for (const key of new Set([...Object.keys(a), ...Object.keys(b), ...Object.keys(c)])) result[key] = merge(a[key], b[key], c[key]);
    return result;
  };
  const next = merge(base, local, fresh), before = base.workspace || {}, live = local.workspace || {}, published = fresh.workspace || {};
  next.workspace ||= {};
  const list = (old = [], current = [], authoritative = []) => authoritative.map((item) => {
    const origin = item.originalReference?.lastBoundReference || item.originalReference || item;
    const match = (entry) => item.instanceId && entry.instanceId === item.instanceId || origin.instanceId && entry.instanceId === origin.instanceId || entry.value === origin.value;
    const user = current.find(match), previous = old.find(match);
    const result = { ...item };
    for (const key of ["weight", "precision", "enabled", "name"]) if (user && previous && !equal(user[key], previous[key])) result[key] = user[key];
    return result;
  });
  if (published.mountedLorasByEngine) {
    next.workspace.mountedLorasByEngine = clone(published.mountedLorasByEngine);
    for (const [engine, members] of Object.entries(published.mountedLorasByEngine.byEngine || {})) next.workspace.mountedLorasByEngine.byEngine[engine] = list(before.mountedLorasByEngine?.byEngine?.[engine], live.mountedLorasByEngine?.byEngine?.[engine], members);
    next.workspace.loras = clone(next.workspace.mountedLorasByEngine.byEngine?.[next.workspace.model] || []);
  } else next.workspace.loras = list(before.loras, live.loras, published.loras);
  if (published.loraGroupsByEngine) {
    next.workspace.loraGroupsByEngine = clone(published.loraGroupsByEngine);
    for (const [engine, groups] of Object.entries(published.loraGroupsByEngine.byEngine || {})) next.workspace.loraGroupsByEngine.byEngine[engine] = groups.map((group) => {
      const old = before.loraGroupsByEngine?.byEngine?.[engine]?.find((item) => item.id === group.id), current = live.loraGroupsByEngine?.byEngine?.[engine]?.find((item) => item.id === group.id);
      return { ...group, ...(current ? merge(old || {}, current, group) : {}), id: group.id, members: list(old?.members, current?.members, group.members) };
    });
  }
  return next;
}

export function rebasePublishedCards(base, local, fresh) {
  const next = clone(fresh), fields = ["title", "prompt", "note", "tags", "cover"];
  for (const [engine, scope] of Object.entries(local?.byEngine || {})) {
    const destination = next.byEngine?.[engine];
    if (!destination) continue;
    for (const [value, card] of Object.entries(scope.loras || {})) {
      const previous = base?.byEngine?.[engine]?.loras?.[value] || {};
      let target = card.instanceId && destination.byInstance?.[card.instanceId];
      if (!target) target = Object.values(destination.byInstance || {}).find((entry) => (entry.originalReference?.lastBoundReference || entry.originalReference)?.value === value);
      target ||= destination.loras?.[value];
      if (!target) continue;
      for (const field of fields) if (!equal(card[field], previous[field])) target[field] = clone(card[field]);
      if (target.instanceId) {
        destination.byInstance[target.instanceId] = target;
        for (const [key, entry] of Object.entries(destination.loras)) if (entry.instanceId === target.instanceId) destination.loras[key] = target;
      }
    }
  }
  return next;
}
