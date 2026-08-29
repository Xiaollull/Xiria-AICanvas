import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import { createUpdateRestartHandoff } from "./update-restart.mjs";

test("supervised update restart exits with the supervisor restart code once", () => {
  const exitCodes = [];
  const handOff = createUpdateRestartHandoff({
    supervised: true,
    projectRoot: path.resolve("test-project"),
    spawnProcess: () => assert.fail("supervised restart must not launch another supervisor"),
    exitProcess: (code) => exitCodes.push(code),
  });

  assert.equal(handOff(), true);
  assert.equal(handOff(), false);
  assert.deepEqual(exitCodes, [77]);
});

test("standalone update restart launches a detached supervisor before exiting", () => {
  const projectRoot = path.resolve("test-project");
  const environment = { TEST_ENVIRONMENT: "1" };
  const calls = [];
  let unrefCount = 0;
  const exitCodes = [];
  const handOff = createUpdateRestartHandoff({
    supervised: false,
    projectRoot,
    environment,
    nodePath: "node-test",
    spawnProcess: (...args) => {
      calls.push(args);
      return { unref: () => { unrefCount += 1; } };
    },
    exitProcess: (code) => exitCodes.push(code),
  });

  assert.equal(handOff(), true);
  assert.equal(handOff(), false);
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], "node-test");
  assert.deepEqual(calls[0][1], [path.join(projectRoot, "scripts", "start.mjs"), "--no-open"]);
  assert.deepEqual(calls[0][2], {
    cwd: projectRoot,
    env: environment,
    detached: true,
    stdio: "ignore",
    windowsHide: true,
  });
  assert.equal(unrefCount, 1);
  assert.deepEqual(exitCodes, [0]);
});
