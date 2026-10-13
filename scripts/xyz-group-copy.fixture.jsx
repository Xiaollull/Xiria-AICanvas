// QPlan2/P2.1 browser fixture: the real App plus a scripted stand-in for the authoritative resolve route.
// Names and hashes are synthetic; nothing here reads a real library, group or model.
await import("./xyz-full-ui-stage1.fixture.jsx");
const stage1 = window.__appFetch, SHARED = "5738b73b9323";
const hex = (n) => (n % 255 + 1).toString(16).padStart(2, "0").repeat(32);
const sum = (text) => [...text].reduce((total, char) => (total * 31 + char.charCodeAt(0)) % 1000003, 7);
const uuid = (value) => `00000000-0000-4000-8000-${String(sum(value)).padStart(12, "0")}`;
const baseName = (value) => value.split("/").at(-1).replace(".safetensors", "");
// Equal basenames share bytes across roots; the stage-1 catalog portraits keep their own hashes.
const hashFor = (value) => { const portrait = /^Anima-portrait-(\d)$/.exec(baseName(value)); return portrait ? String(Number(portrait[1]) + 1).repeat(64) : hex(sum(baseName(value))); };
const legacy = (value, name, extra = {}) => ({ value, name, category: "style", weight: 0.6, precision: 4, enabled: true, ...extra });
const shared = (name, extra) => legacy(`shared:${SHARED}/Anima/Style/${name}.safetensors`, name, extra);
const local = (name, extra) => legacy(`style/NekoX/${name}.safetensors`, name, extra);
const names = ["ashima_v4", "yusano_v5_step1500", "kiriaina_anima_style-000030", "nk_four", "nk_five", "nk_six", "nk_seven", "nk_eight"];
window.__libraryGroups = [
  { id: "g-nkstyle", name: "NKStyleV2", enabled: false, collecting: false, presetPrompt: "nk style words", members: names.map((name, i) => shared(name, i === 3 ? { enabled: false } : i === 5 ? { weight: 0 } : i === 6 ? { weight: -1.2345, precision: 2 } : {})) },
  { id: "g-neko", name: "NekoX V2", enabled: false, collecting: false, presetPrompt: "", members: names.slice(0, 3).map((name) => local(name, { weight: 0.4, precision: 2 })) },
  { id: "g-broken", name: "Broken refs", enabled: false, collecting: false, presetPrompt: "", members: [shared("fine_one"), shared("ambiguous_two"), shared("never_registered"), shared("fine_four")] },
  { id: "g-copies", name: "Portrait copies", enabled: false, collecting: false, presetPrompt: "", members: [shared("Anima-portrait-0", { weight: 0.3 }), shared("Anima-portrait-1", { weight: 0.4 })] },
];
window.__resolveCalls = [];
window.__resolve = { gate: null, revision: 5, ambiguous: new Set(["ambiguous_two"]), fallback: new Set(["never_registered"]), drift: null };
const json = (payload, status = 200, headers = {}) => new Response(JSON.stringify(payload), { status, headers: { "Content-Type": "application/json", ...headers } });
window.__appFetch = async (url, method, body) => {
  if (url.startsWith("/api/lora-relink/resolve")) {
    const state = window.__resolve, reference = body.references[0];
    window.__resolveCalls.push(structuredClone(body));
    if (state.gate) await state.gate;
    const name = baseName(reference.value);
    if (state.ambiguous.has(name)) return json({ error: "LoRA reference is ambiguous-path; choose an explicit target", resolution: { status: "ambiguous-path" } }, 409);
    if (state.fallback.has(name)) return json({ revision: state.revision, references: [{ ...reference, status: "unresolved", promptRewriteAllowed: false }], loras: [] });
    // `drift` simulates a library change after the preview: later resolutions of one name bind another instance.
    const drifted = Boolean(state.drift) && window.__resolveCalls.length > state.drift.after && name === state.drift.name;
    const instanceId = reference.instanceId || uuid(reference.value), contentHash = reference.contentHash || hashFor(reference.value);
    return json({ revision: state.revision, references: [{ ...reference, status: "resolved", value: reference.value, instanceId: drifted ? uuid(`${reference.value}#drift`) : instanceId, contentHash, contentRevision: 2,
      kind: "current", originalValue: reference.value, auditKind: reference.instanceId ? "instance" : "recorded-path", candidates: [], promptRewriteAllowed: false }], loras: [] });
  }
  const answer = await stage1(url, method, body);
  if (url.startsWith("/api/ui-state") && method !== "PUT" && answer) {
    const data = await answer.clone().json();
    data.state.workspace.loraGroupsByEngine.byEngine.Anima = structuredClone(window.__libraryGroups);
    return json(data, 200, { "X-XirAI-LoRA-Revision": "0", "X-XirAI-Document-Version": "fixture-only" });
  }
  return answer;
};
