import { libraryError } from "./lora-library.mjs";

// QPlan2/P1 Phase B. The two XYZ routes that admit work: preview and submission. Status,
// controls and continuation are ordinary proxied reads/commands; continuation re-admits every
// cell on the Python plane under the same lease before it runs.
const ADMITTING_ROUTES = new Set(["/api/inference/xyz-plots", "/api/inference/xyz-plots/plan"]);
const record = (value) => value && typeof value === "object" && !Array.isArray(value);

/** Every LoRA reference a draft could execute, once each. Malformed structured values are left
 * to the Python planner, which reports them with their coordinate instead of a bare 400. */
export async function xyzLoraReferences(draft, service) {
  // The Python planner owns full/overlay/decimal/preset/replacement semantics. Its private stdin
  // projection and real Node resolution share one publication lease; no parallel JS compiler.
  return service.resolveXYZ(draft);
}

/** Runs AFTER the backend identity gate, like the ordinary job route. Node resolves every
 * reference under its lease; an experiment freezes exact identities, so a reference that a repair
 * has relocated is refused for an explicit Sync rather than silently rebound. Python then
 * reacquires the same lease, compares the stamped revision and validates full bytes per cell. */
export function xyzPlotMiddleware({ service, readJson, admit, forward }) {
  return async (request, response, next) => {
    const pathname = new URL(request.url, "http://localhost").pathname;
    if (request.method !== "POST" || !ADMITTING_ROUTES.has(pathname)) { next(); return; }
    try {
      admit(request);
      const payload = await readJson(request);
      if (!record(payload)) throw libraryError("Invalid XYZ request", 400);
      const draft = pathname.endsWith("/plan") ? payload : payload.draft;
      if (!record(draft) || typeof draft.engine !== "string") throw libraryError("Invalid XYZ draft", 400);
      const headers = { ...request.headers };
      delete headers["x-xirai-lora-context"];
      delete headers["x-xirai-lora-chain"];
      const resolved = await xyzLoraReferences(draft, service);
      const references = resolved.effectiveReferences;
      if (references.length) {
        if (resolved.revision && headers["x-xirai-lora-revision"] !== String(resolved.revision)) throw libraryError("LoRA published revision changed; refetch before submitting");
        const moved = resolved.references.findIndex((item, index) => item.value !== references[index]?.value);
        if (moved >= 0 || resolved.references.length !== references.length) {
          throw libraryError(`XYZ 冻结的 LoRA「${references[Math.max(0, moved)]?.value}」已被修复重定位或替换；请在 XYZ 工作区明确同步 LoRA 后再提交`);
        }
        headers["x-xirai-lora-revision"] = String(resolved.revision);
      }
      await forward(pathname, payload, headers, request, response);
    } catch (error) {
      if (response.headersSent) { response.end(); return; }
      response.statusCode = error.statusCode || 503;
      response.setHeader("Content-Type", "application/json; charset=utf-8");
      response.setHeader("Cache-Control", "no-store");
      response.end(JSON.stringify({ detail: error.statusCode ? error.message : "LoRA admission unavailable; XYZ draft retained" }));
    }
  };
}
