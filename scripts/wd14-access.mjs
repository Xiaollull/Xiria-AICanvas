// The inference identity gate is not browser authorization. Keep local filesystem
// operations behind a loopback *socket* and a loopback Host (DNS rebinding guard).
const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]"]);
const LOCAL_PEERS = new Set(["127.0.0.1", "::1", "::ffff:127.0.0.1"]);

export function wd14Admission(request) {
  let pathname, parsed;
  try { parsed = new URL(request.url, "http://localhost"); pathname = decodeURIComponent(parsed.pathname); }
  catch { return { allowed: false, status: 400 }; }
  if (!pathname.startsWith("/api/inference/wd14/")) return { allowed: true };
  const models = pathname === "/api/inference/wd14/models";
  // Presence, not a second bool parser: true/1/on/yes and encoded names all
  // take the privileged path. Duplicate values cannot disagree with Python.
  const catalogRefresh = models && parsed.searchParams.has("refresh");
  if (catalogRefresh && parsed.searchParams.getAll("refresh").length !== 1) return { allowed: false, status: 400 };
  if (request.method !== "POST" && !catalogRefresh) return { allowed: true };
  let origin, host;
  try {
    origin = new URL(request.headers.origin);
    host = new URL(`http://${request.headers.host}`);
  } catch { return { allowed: false, status: 403 }; }
  if (!["http:", "https:"].includes(origin.protocol) || origin.host !== host.host || origin.username || origin.password || origin.pathname !== "/" || origin.search || origin.hash) {
    return { allowed: false, status: 403 };
  }
  if (models || pathname === "/api/inference/wd14/batch" || pathname === "/api/inference/wd14/runtime/install" || pathname.startsWith("/api/inference/wd14/download/") || pathname.endsWith("/cancel")) {
    if (!LOCAL_PEERS.has(request.socket.remoteAddress) || !LOCAL_HOSTS.has(host.hostname)) return { allowed: false, status: 403 };
  }
  return { allowed: true };
}
