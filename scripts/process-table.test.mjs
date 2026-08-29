import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { executableName, isWrapperProcess, parseProcessTable, selectTargets, terminationOrder } from "./process-table.mjs";

const PROJECT = "/home/you/XiriaCanvas AI";

test("ps output parses into the same shape the Windows branch produces", () => {
  const table = parseProcessTable([
    "    1       0 systemd          /sbin/init splash",
    " 4210    4180 MainThread       node /home/you/XiriaCanvas AI/node_modules/vite/bin/vite.js",
    " 4300    4210 python3.12       /home/you/XiriaCanvas AI/.venv/bin/python -I backend/inference_server.py",
    "",
    "garbage line without pids",
  ].join("\n"));
  assert.equal(table.length, 3);
  assert.deepEqual(table[1], {
    pid: 4210,
    parentPid: 4180,
    name: "MainThread",
    commandLine: "node /home/you/XiriaCanvas AI/node_modules/vite/bin/vite.js",
  });
});

test("an interpreter is recognised from the command line when comm cannot name it", () => {
  // Linux reports the *thread* name in `comm`, and Node renames its main thread
  // to "MainThread". Matching on `comm` alone therefore never recognises a Node
  // supervisor on Linux, which is exactly the process cleanup has to reach.
  assert.equal(isWrapperProcess({ name: "MainThread", commandLine: "node scripts/start.mjs" }), true);
  assert.equal(isWrapperProcess({ name: "node.exe", commandLine: "" }), true);
  assert.equal(isWrapperProcess({ name: "python3.12", commandLine: "" }), true);
  assert.equal(isWrapperProcess({ name: "MainThread", commandLine: "" }), false);
  assert.equal(isWrapperProcess({ name: "chrome", commandLine: "/opt/chrome/chrome --app=..." }), false);
  assert.equal(isWrapperProcess(null), false);

  assert.equal(executableName({ commandLine: "node scripts/start.mjs" }), "node");
  assert.equal(executableName({ commandLine: "/usr/bin/python3.12 -I server.py" }), "python3.12");
  // A quoted Windows path may contain spaces; splitting on whitespace would cut it.
  assert.equal(executableName({ commandLine: '"C:\\Program Files\\nodejs\\node.exe" start.mjs' }), "node.exe");
  assert.equal(executableName({ commandLine: "" }), "");
});

// `ps` is the POSIX half of `processTable()`; Windows reads Win32_Process instead.
test("the real host names its own Node process in a way cleanup recognises", { skip: process.platform === "win32" }, () => {
  // Guards the assumption above against a future runtime that reports `comm`
  // differently: whatever this platform calls Node, cleanup must still match it.
  const output = execFileSync("ps", ["-o", "pid=,ppid=,comm=,args=", "-p", String(process.pid)], { encoding: "utf8" });
  const [self] = parseProcessTable(output);
  assert.ok(self, "this platform's ps did not produce a parsable row for the test process");
  assert.equal(self.pid, process.pid);
  assert.equal(isWrapperProcess(self), true, `Node reported itself as ${JSON.stringify(self)}`);
});

test("cleanup selects the workspace tree and the supervisor above it", () => {
  // 4180 is `node scripts/start.mjs`, launched with a relative path, so it is
  // only reachable by climbing from the child that does name the project.
  const processes = [
    { pid: 1, parentPid: 0, name: "systemd", commandLine: "/sbin/init" },
    { pid: 4100, parentPid: 1, name: "bash", commandLine: "sh Start-XirAI.sh" },
    { pid: 4180, parentPid: 4100, name: "MainThread", commandLine: "node scripts/start.mjs" },
    { pid: 4210, parentPid: 4180, name: "MainThread", commandLine: `node ${PROJECT}/node_modules/vite/bin/vite.js` },
    { pid: 4300, parentPid: 4210, name: "python3.12", commandLine: `${PROJECT}/.venv/bin/python backend/inference_server.py` },
    { pid: 9000, parentPid: 1, name: "chrome", commandLine: "/opt/chrome/chrome --app=http://localhost:7709/" },
    { pid: 9100, parentPid: 1, name: "MainThread", commandLine: "node /home/you/other-project/server.js" },
  ];
  const { targets } = selectTargets(processes, { projectRoot: PROJECT, selfPid: 9999 });
  assert.deepEqual([...targets].sort((a, b) => a - b), [4100, 4180, 4210, 4300]);
  // An unrelated Node app and the browser are never touched.
  assert.equal(targets.has(9100), false);
  assert.equal(targets.has(9000), false);
});

test("cleanup never selects itself or the shell that invoked it", () => {
  const processes = [
    { pid: 100, parentPid: 1, name: "bash", commandLine: `bash --login ${PROJECT}` },
    { pid: 200, parentPid: 100, name: "MainThread", commandLine: `node ${PROJECT}/scripts/cleanup-processes.mjs` },
    { pid: 300, parentPid: 1, name: "python3", commandLine: `${PROJECT}/.venv/bin/python backend/inference_server.py` },
  ];
  const { targets } = selectTargets(processes, { projectRoot: PROJECT, selfPid: 200 });
  assert.deepEqual([...targets], [300]);
});

test("children are signalled before the parents that would orphan them", () => {
  const processes = [
    { pid: 10, parentPid: 1, name: "MainThread", commandLine: `node ${PROJECT}/a.js` },
    { pid: 20, parentPid: 10, name: "MainThread", commandLine: `node ${PROJECT}/b.js` },
    { pid: 30, parentPid: 20, name: "python3", commandLine: `${PROJECT}/c.py` },
  ];
  const { targets, byPid } = selectTargets(processes, { projectRoot: PROJECT, selfPid: 999 });
  assert.deepEqual(terminationOrder(targets, byPid), [30, 20, 10]);
});

test("cleanup-processes delegates to the shared table module", async () => {
  const source = await readFile(new URL("./cleanup-processes.mjs", import.meta.url), "utf8");
  assert.match(source, /import \{ parseProcessTable, selectTargets, terminationOrder \} from "\.\/process-table\.mjs"/);
  assert.match(source, /selectTargets\(processTable\(\)\.processes, \{ projectRoot, selfPid: process\.pid \}\)/);
  // The interpreter list must not be re-inlined here; one copy decides.
  assert.doesNotMatch(source, /powershell\|pwsh\|sh\|bash/);
});
