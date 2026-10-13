import { libraryError } from "./lora-library.mjs";

/** Runs AFTER the existing backend identity gate. Node validates mount evidence
 * under its lease; Python independently reacquires that lease and compares this
 * revision through COMPLETE inline preflight + queued admission. A publication
 * in the HTTP gap is a 409, not a check-busy-then-write exception to the gate.
 */
export function loraJobMiddleware({ service, readJson, admit, submit }) {
  return async (request, response, next) => {
    if (new URL(request.url, "http://localhost").pathname !== "/api/inference/jobs" || request.method !== "POST") { next(); return; }
    try {
      admit(request);
      const payload = await readJson(request);
      if (!payload || typeof payload !== "object" || Array.isArray(payload)) throw libraryError("Invalid generation request", 400);
      const headers = { ...request.headers };
      delete headers["x-xirai-lora-chain"];
      if (Array.isArray(payload.loras) && payload.loras.length) {
        let context;
        try { context = JSON.parse(headers["x-xirai-lora-context"] || "[]"); }
        catch { throw libraryError("Invalid LoRA execution context", 400); }
        if (!Array.isArray(context) || context.length > payload.loras.length || payload.loras.length > 16) throw libraryError("Invalid LoRA execution context", 400);
        const refs = payload.loras.map((item, index) => {
          const evidence = context[index];
          if (!item || typeof item.path !== "string" || (evidence !== undefined && (!evidence || typeof evidence !== "object" || Array.isArray(evidence)))) throw libraryError("Invalid LoRA execution row", 400);
          return { value: item.path, weight: item.weight, ...(evidence?.instanceId ? { identitySchema: 1, instanceId: evidence.instanceId } : {}), ...(evidence?.contentHash ? { contentHash: evidence.contentHash } : {}) };
        });
        const resolved = await service.resolve(payload.engine, refs);
        if (resolved.revision && headers["x-xirai-lora-revision"] !== String(resolved.revision)) throw libraryError("LoRA published revision changed; refetch before submitting");
        payload.loras = resolved.loras;
        headers["x-xirai-lora-revision"] = String(resolved.revision);
        headers["x-xirai-lora-context"] = JSON.stringify(resolved.references.map(({ instanceId, contentHash }) => ({ instanceId, contentHash })));
      }
      await submit(payload, headers, request, response);
    } catch (error) {
      if (response.headersSent) { response.end(); return; }
      response.statusCode = error.statusCode || 503;
      response.setHeader("Content-Type", "application/json; charset=utf-8");
      response.setHeader("Cache-Control", "no-store");
      response.end(JSON.stringify({ detail: error.statusCode ? error.message : "LoRA admission unavailable; original job snapshots retained" }));
    }
  };
}
