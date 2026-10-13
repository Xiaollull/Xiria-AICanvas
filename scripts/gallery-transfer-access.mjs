// Gallery import/export writes under the output root and, for an import by path, reads a file on
// this machine. Every transfer write needs a strict same-Origin page; the path import also needs a
// loopback socket and a loopback Host (DNS rebinding guard), like the WD14 folder batch.
const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]"]);
const LOCAL_PEERS = new Set(["127.0.0.1", "::1", "::ffff:127.0.0.1"]);
const TRANSFER_ROUTE = /^\/api\/inference\/gallery\/transfer\/(export|import|import-upload)$/;

function transferRoute(request) {
  try {
    const match = TRANSFER_ROUTE.exec(decodeURIComponent(new URL(request.url, "http://localhost").pathname));
    return request.method === "POST" && match ? match[1] : null;
  } catch { return null; }
}

export function isGalleryTransferRequest(request) {
  return transferRoute(request) !== null;
}

export function galleryTransferAdmission(request) {
  const route = transferRoute(request);
  if (!route) return { allowed: true };
  let origin, host;
  try {
    origin = new URL(request.headers.origin);
    host = new URL(`http://${request.headers.host}`);
  } catch { return { allowed: false, status: 403 }; }
  if (!["http:", "https:"].includes(origin.protocol) || origin.host !== host.host || origin.username || origin.password || origin.pathname !== "/" || origin.search || origin.hash) {
    return { allowed: false, status: 403 };
  }
  if (route === "import" && (!LOCAL_PEERS.has(request.socket?.remoteAddress) || !LOCAL_HOSTS.has(host.hostname))) {
    return { allowed: false, status: 403, local: true };
  }
  return { allowed: true };
}
