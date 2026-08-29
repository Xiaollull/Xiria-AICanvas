import { createHash, randomUUID } from "node:crypto";
import { realpath } from "node:fs/promises";
import { createConnection, createServer } from "node:net";
import os from "node:os";
import path from "node:path";

const activeLocks = new WeakSet();

function normalizedPath(value) {
  const resolved = path.resolve(value);
  return process.platform === "win32" ? resolved.toLowerCase() : resolved;
}

function lockEndpoint(projectRoot) {
  const id = createHash("sha256").update(normalizedPath(projectRoot)).digest("hex").slice(0, 32);
  if (process.platform === "win32") return `\\\\.\\pipe\\xirai-offline-update-${id}`;
  if (process.platform === "linux") return `\0xirai-offline-update-${id}`;
  return path.join(os.tmpdir(), `xirai-offline-update-${id}.sock`);
}

function currentOwner(endpoint) {
  return new Promise((resolve) => {
    let output = "";
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      socket.destroy();
      resolve(value);
    };
    const socket = createConnection(endpoint);
    const timer = setTimeout(() => finish(null), 1500);
    socket.setEncoding("utf8");
    socket.on("data", (chunk) => { output = `${output}${chunk}`.slice(0, 8192); });
    socket.once("end", () => {
      try { finish(JSON.parse(output)); } catch { finish(null); }
    });
    socket.once("error", () => finish(null));
  });
}

export async function acquireOfflineUpdateLock({ projectRoot, operation } = {}) {
  if (!projectRoot) throw new Error("缺少离线更新项目根目录");
  const root = await realpath(path.resolve(projectRoot));
  const endpoint = lockEndpoint(root);
  const owner = {
    schema: 1,
    product: "XiriaCanvas AI",
    projectRoot: root,
    ownerPid: process.pid,
    operation: operation || "offline-update",
    token: randomUUID(),
    createdAt: new Date().toISOString(),
  };
  const server = createServer((socket) => socket.end(`${JSON.stringify(owner)}\n`));
  try {
    await new Promise((resolve, reject) => {
      server.once("error", reject);
      server.listen(endpoint, resolve);
    });
  } catch (error) {
    server.close();
    if (error.code !== "EADDRINUSE") throw error;
    const existing = await currentOwner(endpoint);
    const busy = new Error(existing?.ownerPid
      ? `离线更新正在运行（PID ${existing.ownerPid}），请等待完成后再启动或重试`
      : "离线更新锁已被占用，请等待当前操作完成");
    busy.code = "UPDATE_BUSY";
    busy.statusCode = 409;
    throw busy;
  }

  let released = false;
  const lock = {
    ...owner,
    endpoint,
    async release() {
      if (released) return;
      released = true;
      activeLocks.delete(lock);
      await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
    },
  };
  activeLocks.add(lock);
  return lock;
}

export function assertOfflineUpdateLock(lock, projectRoot) {
  if (!lock || !activeLocks.has(lock) || normalizedPath(lock.projectRoot) !== normalizedPath(projectRoot)) {
    throw new Error("离线更新恢复必须持有当前项目的进程锁");
  }
}
