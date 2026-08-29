import { execFileSync, spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadLocalEnv } from "./python.mjs";
import { acquireOfflineUpdateLock } from "./offline-update-lock.mjs";
import { parseProcessTable, selectTargets, terminationOrder } from "./process-table.mjs";

const projectRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const environment = { ...process.env, ...loadLocalEnv(projectRoot) };
const ports = [environment.WEB_PORT || "7709", environment.INFERENCE_PORT || "8718"]
  .map(Number)
  .filter((port) => Number.isInteger(port) && port > 0 && port <= 65535);
const dryRun = process.argv.includes("--dry-run");
const stateDirectory = path.resolve(projectRoot, environment.XIRAI_STATE_DIR || "state-cache");

function processTable() {
  if (process.platform === "win32") {
    const script = [
      "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8",
      "$processes=@(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine)",
      `$owners=@(Get-NetTCPConnection -State Listen -LocalPort ${ports.join(",")} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique)`,
      "@{processes=$processes;owners=$owners}|ConvertTo-Json -Compress -Depth 4",
    ].join(";");
    const output = execFileSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", script], {
      encoding: "utf8",
      windowsHide: true,
    });
    const payload = JSON.parse(output);
    return {
      processes: [payload.processes || []].flat().map((item) => ({
        pid: Number(item.ProcessId),
        parentPid: Number(item.ParentProcessId),
        name: String(item.Name || ""),
        commandLine: String(item.CommandLine || ""),
      })),
      portOwners: new Set([payload.owners || []].flat().map(Number).filter(Boolean)),
    };
  }

  const processes = parseProcessTable(execFileSync("ps", ["-eo", "pid=,ppid=,comm=,args="], { encoding: "utf8" }));
  const portOwners = new Set();
  const lsof = spawnSync("lsof", ["-nP", "-t", ...ports.map((port) => `-iTCP:${port}`), "-sTCP:LISTEN"], { encoding: "utf8" });
  if (!lsof.error && lsof.status === 0) {
    for (const value of lsof.stdout.split(/\s+/)) {
      if (Number(value)) portOwners.add(Number(value));
    }
  } else {
    for (const port of ports) {
      const fuser = spawnSync("fuser", ["-n", "tcp", String(port)], { encoding: "utf8" });
      for (const value of `${fuser.stdout} ${fuser.stderr}`.match(/\d+/g) || []) portOwners.add(Number(value));
    }
  }
  return { processes, portOwners };
}

function terminate({ targets, byPid }) {
  const ordered = terminationOrder(targets, byPid);
  if (!ordered.length) {
    console.log(`没有检测到 XiriaCanvas AI 残留进程，端口 ${ports.join(" / ")} 已可用于启动。`);
    return;
  }
  console.log(`检测到 ${ordered.length} 个残留进程，正在清理...`);
  for (const pid of ordered) {
    const item = byPid.get(pid);
    console.log(`  ${dryRun ? "[检查]" : "[终止]"} PID ${pid} · ${item?.name || "unknown"}`);
    if (dryRun) continue;
    try {
      process.kill(pid, "SIGTERM");
    } catch (error) {
      if (error.code !== "ESRCH") throw error;
    }
  }
  if (dryRun) return;

  const deadline = Date.now() + 3000;
  let remaining = [];
  do {
    remaining = ordered.filter((pid) => {
      try {
        process.kill(pid, 0);
        return true;
      } catch {
        return false;
      }
    });
    if (!remaining.length || Date.now() >= deadline) break;
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 100);
  } while (true);

  for (const pid of remaining) {
    try {
      process.kill(pid, "SIGKILL");
    } catch (error) {
      if (error.code !== "ESRCH") throw error;
    }
  }
  console.log(`残留进程清理完成，正在启动新实例。`);
}

let updateLock;
try {
  updateLock = await acquireOfflineUpdateLock({ stateDirectory, projectRoot, operation: "startup-cleanup" });
  terminate(selectTargets(processTable().processes, { projectRoot, selfPid: process.pid }));
} catch (error) {
  console.error(`清理残留进程失败：${error.message}`);
  process.exitCode = 1;
} finally {
  await updateLock?.release().catch(() => {});
}
