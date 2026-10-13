import { wd14Admission } from "./wd14-access.mjs";

export function isTaskDeletionRequest(request) {
  try {
    const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    return request.method === "POST" && /^\/api\/inference\/(xyz-plots|lora-mix)\/task-deletions(?:\/preview)?$/.test(pathname);
  } catch { return false; }
}

// Reuse the existing trusted-local filesystem gate, including loopback socket, Host and strict Origin.
export function taskDeletionAdmission(request) {
  return isTaskDeletionRequest(request) ? wd14Admission({ ...request, socket: request.socket, headers: request.headers, url: "/api/inference/wd14/batch" }) : { allowed: true };
}
