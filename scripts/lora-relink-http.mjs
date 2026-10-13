/** Public management routes; injected owners are production service instances,
 * not user-selected directories, executables, SQLite queries or lease tokens.
 */
export function loraRelinkMiddleware({ service, readJson, admit }) {
  return async (request, response, next) => {
    const url = new URL(request.url, "http://localhost");
    if (!url.pathname.startsWith("/api/lora-relink/")) { next(); return; }
    response.setHeader("Content-Type", "application/json; charset=utf-8");
    response.setHeader("Cache-Control", "no-store");
    try {
      const action = url.pathname.slice("/api/lora-relink/".length);
      admit(request, action);
      if (["state", "status"].includes(action) && request.method === "GET") {
        const value = await service.status(); response.end(JSON.stringify(value)); return;
      }
      if (request.method !== "POST") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
      const payload = await readJson(request, 2 * 1024 * 1024);
      const allowed = {
        scan: ["engines", "rootIds", "maxBytes", "source", "target"], scope: ["engines", "source", "target"], preview: ["scanId", "selections", "registerBaseline"],
        commit: ["scanId", "selections", "registerBaseline", "operationId"], undo: ["operationId", "undoOperationId"],
        resolve: ["engine", "references", "history"], recover: [],
      }[action];
      if (!allowed) throw Object.assign(new Error("Unknown LoRA management action"), { statusCode: 404 });
      if (!payload || typeof payload !== "object" || Array.isArray(payload) || Object.keys(payload).some((key) => !allowed.includes(key))) throw Object.assign(new Error("Unexpected management arguments"), { statusCode: 400 });
      let value;
      if (action === "scan") {
        const controller = new AbortController();
        const cancel = () => controller.abort();
        request.once("aborted", cancel);
        response.once("close", () => { if (!response.writableEnded) cancel(); });
        try { value = await service.scan(payload, controller.signal); } finally { request.removeListener("aborted", cancel); }
      } else if (action === "resolve") {
        if (!Array.isArray(payload.references) || payload.references.length > 16) throw Object.assign(new Error("Invalid reference list"), { statusCode: 400 });
        value = await service.resolve(payload.engine, payload.references, { history: payload.history === true });
      } else if (action === "recover") value = await service.coordinator.recover();
      else value = await service[action](payload);
      if (value.revision !== undefined) response.setHeader("X-XirAI-LoRA-Revision", String(value.revision));
      response.end(JSON.stringify(value));
    } catch (error) {
      response.statusCode = error.statusCode || 503;
      response.end(JSON.stringify({ error: error.statusCode ? error.message : "LoRA management unavailable; original data and recovery evidence retained",
        ...(error.conflicts ? { conflicts: error.conflicts } : {}), ...(error.resolution ? { resolution: error.resolution } : {}) }));
    }
  };
}
