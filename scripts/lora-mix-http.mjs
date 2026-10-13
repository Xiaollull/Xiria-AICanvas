import { libraryError } from "./lora-library.mjs";

// QPlan2/P2. The two search routes that admit work get the same Node LoRA admission as XYZ and ordinary jobs
// before Python sees them: every reference the round could fuse (the frozen global background plus every test
// member) is resolved under the publication lease, a moved/replaced reference is refused for an explicit Sync,
// and Python then re-reads the stamped revision and verifies full bytes per cell. Reads, the used-vector query,
// sessions and release are ordinary proxied calls.
const ADMITTING_ROUTES = new Set(["/api/inference/lora-mix/plan", "/api/inference/lora-mix/sessions"]);
const record = (value) => value && typeof value === "object" && !Array.isArray(value);

/** Every reference a round can execute: enabled globals, then every test member. Malformed shapes are left to Python. */
export function loraMixReferences(draft) {
  if (!record(draft?.baseline) || !record(draft?.mix)) return null;
  const globals = Array.isArray(draft.baseline.loras) ? draft.baseline.loras : [], members = Array.isArray(draft.mix.members) ? draft.mix.members : [];
  if (!globals.every(record) || !members.every((member) => record(member) && record(member.ref) && typeof member.ref.value === "string")) return null;
  return [...globals.filter((item) => item.enabled !== false && typeof item.value === "string"), ...members.map((member) => ({ ...member.ref, weight: "1", enabled: true }))];
}

export function loraMixMiddleware({ service, readJson, admit, forward }) {
  return async (request, response, next) => {
    const pathname = new URL(request.url, "http://localhost").pathname;
    if (request.method !== "POST" || !ADMITTING_ROUTES.has(pathname)) { next(); return; }
    try {
      admit(request);
      const payload = await readJson(request);
      if (!record(payload) || !record(payload.draft) || typeof payload.draft.engine !== "string") throw libraryError("Invalid LoRA mix request", 400);
      const headers = { ...request.headers };
      delete headers["x-xirai-lora-context"];
      delete headers["x-xirai-lora-chain"];
      const references = loraMixReferences(payload.draft);
      if (references?.length) {
        const resolved = await service.resolveReferences(payload.draft.engine, references);
        if (resolved.revision && headers["x-xirai-lora-revision"] !== String(resolved.revision)) throw libraryError("LoRA published revision changed; refetch before submitting");
        const moved = resolved.references.findIndex((item, index) => item.value !== references[index]?.value);
        if (moved >= 0 || resolved.references.length !== references.length) {
          throw libraryError(`搜索冻结的 LoRA「${references[Math.max(0, moved)]?.value}」已被修复重定位或替换；请在搜索工作区明确同步 LoRA 后再提交`);
        }
        headers["x-xirai-lora-revision"] = String(resolved.revision);
      }
      await forward(pathname, payload, headers, request, response);
    } catch (error) {
      if (response.headersSent) { response.end(); return; }
      response.statusCode = error.statusCode || 503;
      response.setHeader("Content-Type", "application/json; charset=utf-8");
      response.setHeader("Cache-Control", "no-store");
      response.end(JSON.stringify({ detail: error.statusCode ? error.message : "LoRA admission unavailable; search draft retained" }));
    }
  };
}
