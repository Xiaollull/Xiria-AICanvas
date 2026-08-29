import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import { createConfiguratorHandoff } from "./configurator-handoff.mjs";

test("supervised configurator handoff exits with the supervisor switch code once", () => {
  const exitCodes = [];
  const handOff = createConfiguratorHandoff({
    supervised: true,
    projectRoot: path.resolve("test-project"),
    spawnProcess: () => assert.fail("supervised handoff must not launch another configurator"),
    exitProcess: (code) => exitCodes.push(code),
  });

  assert.equal(handOff(), true);
  assert.equal(handOff(), false);
  assert.deepEqual(exitCodes, [75]);
});

test("standalone configurator handoff launches a detached configurator before exiting", () => {
  const projectRoot = path.resolve("test-project");
  const environment = { TEST_ENVIRONMENT: "1" };
  const calls = [];
  let unrefCount = 0;
  const exitCodes = [];
  const handOff = createConfiguratorHandoff({
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
  assert.deepEqual(calls[0][1], [path.join(projectRoot, "scripts", "setup-gui.mjs"), "--no-open", "--return-to-app"]);
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
