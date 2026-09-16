import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const page = await readFile(new URL("../src/ManualUpdatePage.jsx", import.meta.url), "utf8");

test("remote offline update authorization stays in page memory and reaches every mutating update request", () => {
  assert.match(page, /const \[remoteUpdateToken, setRemoteUpdateToken\] = useState\(""\);/);
  assert.match(page, /X-XirAI-Remote-Update-Token/);
  assert.match(page, /remote_offline_update_enabled/);
  for (const route of ["archive", "apply", "repair", "restart"]) {
    assert.match(page, new RegExp(`/api/system/update/${route}`));
  }
  const tokenState = page.slice(page.indexOf("const [remoteUpdateToken"), page.indexOf("const restartReturnStarted"));
  assert.doesNotMatch(tokenState, /localStorage|sessionStorage|fetch\(/, "the remote update secret is not persisted or sent during initialization");
});

test("the update key reaches the web server and nothing else it starts", async () => {
  const { isolatedPythonEnv, withoutUpdateSecrets, UPDATE_SECRET_VARIABLES } = await import("./python.mjs");
  const environment = { PATH: "/usr/bin", XIRAI_REMOTE_UPDATE_TOKEN: "a".repeat(64), XIRAI_UPDATE_TOKEN: "ghp_example", KEEP: "1" };
  for (const build of [isolatedPythonEnv, withoutUpdateSecrets]) {
    const child = build(environment);
    for (const name of UPDATE_SECRET_VARIABLES) assert.equal(name in child, false, `${build.name} must not pass ${name} to a child`);
    assert.equal(child.PATH, "/usr/bin", `${build.name} keeps the rest of the environment`);
    assert.equal(child.KEEP, "1");
  }
  // Building a child environment must not mutate the parent's.
  assert.equal(environment.XIRAI_REMOTE_UPDATE_TOKEN, "a".repeat(64));

  const start = await readFile(new URL("../scripts/start.mjs", import.meta.url), "utf8");
  // The configurator installs an environment and never authorizes an update.
  assert.match(start, /scripts", "setup-gui\.mjs"[\s\S]{0,400}env: withoutUpdateSecrets\(process\.env\)/);
  // The main service is the one that checks the key, so it is the one that still receives it.
  assert.match(start, /env: \{ \.\.\.process\.env, XIRAI_SERVICE_SUPERVISOR: "1" \}/);
  assert.doesNotMatch(start, /env: withoutUpdateSecrets\(\{ \.\.\.process\.env, XIRAI_SERVICE_SUPERVISOR/, "stripping the key from the web server would disable remote updates entirely");
});

test("the documented key shape matches the one the server accepts", async () => {
  const [config, example] = await Promise.all([
    readFile(new URL("../vite.config.js", import.meta.url), "utf8"),
    readFile(new URL("../.env.example", import.meta.url), "utf8"),
  ]);
  assert.match(config, /REMOTE_UPDATE_TOKEN_PATTERN = \/\^\[a-f0-9\]\{64\}\$\/i/);
  // The sample file used to promise "32+ characters", which the server refuses.
  assert.doesNotMatch(example, /32\+ character/);
  assert.match(example, /exactly 64 hexadecimal characters/);
  assert.match(example, /openssl rand -hex 32/);
  assert.match(example, /XIRAI_REMOTE_UPDATE_ALLOW_INSECURE/);
  assert.match(example, /required from every caller, including localhost/);
  const page = await readFile(new URL("../src/ManualUpdatePage.jsx", import.meta.url), "utf8");
  // The page used to tell the user a local update never needs the key, which stopped being true.
  assert.doesNotMatch(page, /实例本机更新无需填写。/);
  assert.match(page, /包括本机在内/);
});
