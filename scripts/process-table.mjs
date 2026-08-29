// Reading the process tree is the one part of startup cleanup where the two
// platforms disagree about what a process is even called, so it lives here
// where it can be tested without enumerating the real machine.
import path from "node:path";

export function normalizedPath(value) {
  const resolved = path.normalize(String(value || ""));
  return process.platform === "win32" ? resolved.toLowerCase() : resolved;
}

// `ps -eo pid=,ppid=,comm=,args=`. `comm` is whitespace-free, so the first
// three columns are unambiguous and everything after them is the command line.
export function parseProcessTable(output) {
  return String(output || "").split(/\r?\n/).flatMap((line) => {
    const match = line.trim().match(/^(\d+)\s+(\d+)\s+(\S+)\s*(.*)$/);
    return match ? [{ pid: Number(match[1]), parentPid: Number(match[2]), name: match[3], commandLine: match[4] }] : [];
  });
}

const WRAPPER_PROCESS = /^(node|python|python\d+(?:\.\d+)?|cmd|powershell|pwsh|sh|bash)(?:\.exe)?$/i;

// The executable a process was launched as. Windows reports it in `Name`
// already; on Linux `comm` is the *thread* name, which Node overwrites with
// "MainThread", so the command line is the only place the interpreter is still
// visible. A quoted first token is a Windows-style path that may contain
// spaces ("C:\Program Files\nodejs\node.exe").
export function executableName(item) {
  const command = String(item?.commandLine || "").trim();
  if (!command) return "";
  const quoted = command.startsWith('"') ? command.slice(1, command.indexOf('"', 1)) : "";
  const first = quoted || command.split(/\s+/)[0] || "";
  return first ? path.basename(first.split("\\").join("/")) : "";
}

// True for the interpreters and shells this app launches itself through, so
// that killing a service also kills the supervisor that would otherwise be
// left holding a half-torn-down workspace. Either spelling counts: neither
// source is trustworthy alone across both platforms.
export function isWrapperProcess(item) {
  if (!item) return false;
  return WRAPPER_PROCESS.test(String(item.name || "")) || WRAPPER_PROCESS.test(executableName(item));
}

// Every process that belongs to this workspace: the ones whose command line
// names the project, their descendants, and the interpreter chain above them.
// The caller's own process and its ancestors are never selected — cleanup must
// not kill the shell that invoked it.
export function selectTargets(processes, { projectRoot, selfPid }) {
  const byPid = new Map(processes.map((item) => [item.pid, item]));
  const projectNeedle = normalizedPath(projectRoot);
  const excluded = new Set([selfPid]);
  let ancestor = byPid.get(selfPid)?.parentPid;
  while (ancestor && !excluded.has(ancestor)) {
    excluded.add(ancestor);
    ancestor = byPid.get(ancestor)?.parentPid;
  }

  const targets = new Set();
  for (const item of processes) {
    if (excluded.has(item.pid)) continue;
    if (normalizedPath(item.commandLine).includes(projectNeedle)) targets.add(item.pid);
  }

  let changed = true;
  while (changed) {
    changed = false;
    for (const item of processes) {
      if (excluded.has(item.pid)) continue;
      if (targets.has(item.parentPid) && !targets.has(item.pid)) {
        targets.add(item.pid);
        changed = true;
      }
    }
  }

  for (const pid of [...targets]) {
    let parentPid = byPid.get(pid)?.parentPid;
    while (parentPid && !excluded.has(parentPid)) {
      const parent = byPid.get(parentPid);
      if (!isWrapperProcess(parent)) break;
      targets.add(parentPid);
      parentPid = parent.parentPid;
    }
  }
  return { targets, byPid, excluded };
}

// Deepest first, so a parent is never signalled before the children it would
// otherwise orphan.
export function terminationOrder(targets, byPid) {
  const depth = (pid) => {
    let value = 0;
    let parentPid = byPid.get(pid)?.parentPid;
    while (targets.has(parentPid)) {
      value += 1;
      parentPid = byPid.get(parentPid)?.parentPid;
    }
    return value;
  };
  return [...targets].sort((first, second) => depth(second) - depth(first));
}
