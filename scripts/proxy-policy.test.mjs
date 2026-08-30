import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createHttpFetch, npmInvocation } from "./node-tools.mjs";
import {
  applyInstallerProxyPolicy,
  createInstallerFetch,
  installerEnvironment,
  npmInstallerEnvironment,
  proxyOptIn,
} from "./proxy-policy.mjs";

const projectRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve(`http://127.0.0.1:${server.address().port}`);
    });
  });
}

function close(server) {
  return new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
}

function runNode(script, environment) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ["--input-type=module", "-e", script], {
      cwd: projectRoot,
      env: environment,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.once("error", reject);
    child.once("exit", (status) => resolve({ status, stdout, stderr }));
  });
}

test("installer defaults to a direct-only child environment", () => {
  const source = {
    HTTP_PROXY: "http://vpn.example:7890",
    https_proxy: "http://vpn.example:7890",
    ALL_PROXY: "socks5://vpn.example:7890",
    NPM_CONFIG_PROXY: "http://vpn.example:7890",
    npm_config_https_proxy: "http://vpn.example:7890",
    PIP_PROXY: "http://vpn.example:7890",
    UV_HTTP_PROXY: "http://vpn.example:7890",
    NODE_USE_ENV_PROXY: "1",
    NODE_OPTIONS: "--trace-warnings --use-env-proxy",
    NO_PROXY: "localhost,127.0.0.1",
    no_proxy: "::1",
    PATH: "test-path",
  };
  const result = installerEnvironment(source, { useProxy: false });
  for (const name of [
    "HTTP_PROXY", "https_proxy", "ALL_PROXY", "NPM_CONFIG_PROXY",
    "npm_config_https_proxy", "PIP_PROXY", "UV_HTTP_PROXY", "NODE_USE_ENV_PROXY",
  ]) {
    assert.equal(result[name], undefined, `${name} must not reach an installer child`);
  }
  assert.equal(result.NODE_OPTIONS, "--trace-warnings");
  assert.equal(result.NO_PROXY, "localhost,127.0.0.1");
  assert.equal(result.no_proxy, "::1");
  assert.equal(result.PATH, "test-path");
  assert.equal(source.HTTP_PROXY, "http://vpn.example:7890", "the caller environment is not mutated");
});

test("explicit proxy opt-in preserves proxy values and ALL_PROXY supplies HTTP(S)", () => {
  const source = { ALL_PROXY: "http://proxy.example:7890", NO_PROXY: "localhost" };
  assert.equal(proxyOptIn(["--use-proxy"], source), true);
  assert.equal(proxyOptIn([], { XIRAI_USE_PROXY: "1" }), true);
  const result = installerEnvironment(source, { useProxy: true });
  assert.equal(result.ALL_PROXY, source.ALL_PROXY);
  assert.equal(result.HTTP_PROXY, source.ALL_PROXY);
  assert.equal(result.HTTPS_PROXY, source.ALL_PROXY);
  assert.equal(result.NO_PROXY, "localhost");
});

test("applying direct policy removes startup proxy toggles from this process environment", () => {
  const environment = {
    HTTP_PROXY: "http://vpn.example:7890",
    PIP_PROXY: "http://vpn.example:7890",
    NODE_USE_ENV_PROXY: "1",
    NODE_OPTIONS: "--use-env-proxy --trace-deprecation",
    NO_PROXY: "localhost",
  };
  const policy = applyInstallerProxyPolicy([], environment);
  assert.equal(policy.useProxy, false);
  assert.equal(policy.configured, false);
  assert.equal(environment.HTTP_PROXY, undefined);
  assert.equal(environment.PIP_PROXY, undefined);
  assert.equal(environment.NODE_USE_ENV_PROXY, undefined);
  assert.equal(environment.NODE_OPTIONS, "--trace-deprecation");
  assert.equal(environment.NO_PROXY, "localhost");
});

test("direct adapter bypasses a configured proxy while explicit mode uses it", async (context) => {
  let targetHits = 0;
  let proxyHits = 0;
  const target = createServer((_request, response) => {
    targetHits += 1;
    response.end("target");
  });
  const proxy = createServer((request, response) => {
    proxyHits += 1;
    assert.match(request.url, /^http:\/\/127\.0\.0\.1:\d+\/fixture$/);
    response.end("proxy");
  });
  const [targetUrl, proxyUrl] = await Promise.all([listen(target), listen(proxy)]);
  context.after(async () => Promise.all([close(target), close(proxy)]));
  const hostileEnvironment = {
    HTTP_PROXY: proxyUrl,
    HTTPS_PROXY: proxyUrl,
    ALL_PROXY: proxyUrl,
    NO_PROXY: "",
  };

  const direct = createInstallerFetch({ useProxy: false, environment: hostileEnvironment });
  assert.equal(await (await direct(`${targetUrl}/fixture`)).text(), "target");
  assert.equal(targetHits, 1);
  assert.equal(proxyHits, 0);

  const optedIn = createInstallerFetch({ useProxy: true, environment: hostileEnvironment });
  assert.equal(await (await optedIn(`${targetUrl}/fixture`)).text(), "proxy");
  assert.equal(targetHits, 1, "the proxy response must not be mistaken for a direct request");
  assert.equal(proxyHits, 1);
});

test("direct adapter remains direct when Node enabled ambient env proxying at startup", async (context) => {
  let targetHits = 0;
  let proxyHits = 0;
  const target = createServer((_request, response) => {
    targetHits += 1;
    response.end("child-target");
  });
  const proxy = createServer((_request, response) => {
    proxyHits += 1;
    response.end("child-proxy");
  });
  const [targetUrl, proxyUrl] = await Promise.all([listen(target), listen(proxy)]);
  context.after(async () => Promise.all([close(target), close(proxy)]));
  const moduleUrl = new URL("./proxy-policy.mjs", import.meta.url).href;
  const script = [
    `const { createInstallerFetch } = await import(${JSON.stringify(moduleUrl)});`,
    "const direct = createInstallerFetch({ useProxy: false, environment: process.env });",
    `const response = await direct(${JSON.stringify(`${targetUrl}/ambient`)});`,
    "console.log(await response.text());",
  ].join("\n");
  const result = await runNode(script, {
    ...process.env,
    HTTP_PROXY: proxyUrl,
    HTTPS_PROXY: proxyUrl,
    ALL_PROXY: proxyUrl,
    NO_PROXY: "",
    NODE_USE_ENV_PROXY: "1",
    NODE_OPTIONS: "--use-env-proxy",
  });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.trim(), "child-target");
  assert.equal(targetHits, 1);
  assert.equal(proxyHits, 0);
});

test("an explicitly requested but missing proxy fails instead of falling back to direct", async () => {
  const fetcher = createInstallerFetch({ useProxy: true, environment: { NO_PROXY: "localhost" } });
  await assert.rejects(fetcher("https://example.invalid/"), /已显式启用代理，但未设置/);
});

test("npm direct boundary overrides hostile user/global npmrc in an actual child", (context) => {
  const temporaryDirectory = mkdtempSync(path.join(os.tmpdir(), "xirai-npm-boundary-"));
  context.after(() => rmSync(temporaryDirectory, { recursive: true, force: true }));
  const userConfig = path.join(temporaryDirectory, "controlled.npmrc");
  const globalConfig = path.join(temporaryDirectory, "hostile-global.npmrc");
  writeFileSync(userConfig, "proxy=null\nhttps-proxy=null\n", "utf8");
  writeFileSync(globalConfig, "proxy=http://global.invalid:8765\nhttps-proxy=http://global.invalid:8765\n", "utf8");
  const environment = npmInstallerEnvironment({
    ...process.env,
    HTTP_PROXY: "http://ambient.invalid:7890",
    HTTPS_PROXY: "http://ambient.invalid:7890",
    NPM_CONFIG_GLOBALCONFIG: globalConfig,
  }, { useProxy: false, userConfig });
  const npm = npmInvocation();
  const readConfig = (name) => spawnSync(npm.command, [...npm.args, "config", "get", name], {
    cwd: projectRoot,
    env: environment,
    encoding: "utf8",
    windowsHide: true,
  });

  const proxy = readConfig("proxy");
  const httpsProxy = readConfig("https-proxy");
  const configuredUserFile = readConfig("userconfig");
  assert.equal(proxy.status, 0, proxy.stderr);
  assert.equal(httpsProxy.status, 0, httpsProxy.stderr);
  assert.equal(configuredUserFile.status, 0, configuredUserFile.stderr);
  assert.equal(proxy.stdout.trim(), "null");
  assert.equal(httpsProxy.stdout.trim(), "null");
  assert.equal(path.resolve(configuredUserFile.stdout.trim()), path.resolve(userConfig));
});

test("the project npmrc outranks the npm_config_userconfig every npm run exports", (context) => {
  const temporaryDirectory = mkdtempSync(path.join(os.tmpdir(), "xirai-npm-userconfig-"));
  context.after(() => rmSync(temporaryDirectory, { recursive: true, force: true }));
  const userConfig = path.join(temporaryDirectory, "controlled.npmrc");
  const inheritedConfig = path.join(temporaryDirectory, "inherited.npmrc");
  writeFileSync(userConfig, "proxy=null\nhttps-proxy=null\n", "utf8");
  writeFileSync(inheritedConfig, "proxy=http://inherited.invalid:9999\n", "utf8");

  // `npm run setup` is how the wizard is started, so setup.mjs always inherits a
  // lowercase npm_config_userconfig. It is listed after the uppercase spelling here
  // because npm keeps whichever name it reads last, and that ordering is the one that
  // used to hand the user's own npmrc — proxy line and all — back to the installer.
  const environment = npmInstallerEnvironment({
    ...process.env,
    NPM_CONFIG_USERCONFIG: path.join(temporaryDirectory, "stale.npmrc"),
    npm_config_userconfig: inheritedConfig,
  }, { useProxy: false, userConfig });
  assert.equal(environment.npm_config_userconfig, undefined, "the inherited spelling must be removed");

  const npm = npmInvocation();
  const readConfig = (name) => spawnSync(npm.command, [...npm.args, "config", "get", name], {
    cwd: projectRoot,
    env: environment,
    encoding: "utf8",
    windowsHide: true,
  });
  const configuredUserFile = readConfig("userconfig");
  const proxy = readConfig("proxy");
  assert.equal(configuredUserFile.status, 0, configuredUserFile.stderr);
  assert.equal(proxy.status, 0, proxy.stderr);
  assert.equal(path.resolve(configuredUserFile.stdout.trim()), path.resolve(userConfig));
  assert.equal(proxy.stdout.trim(), "null");
});

test("proxy credentials reach the proxy only, never the origin server", async (context) => {
  const received = [];
  const target = createServer((_request, response) => response.end("target"));
  const proxy = createServer((request, response) => {
    received.push(request.headers);
    response.end("proxy");
  });
  const [targetUrl, proxyUrl] = await Promise.all([listen(target), listen(proxy)]);
  context.after(async () => Promise.all([close(target), close(proxy)]));
  const credentialled = new URL(proxyUrl);
  credentialled.username = "operator";
  credentialled.password = "s3cret";

  const optedIn = createInstallerFetch({
    useProxy: true,
    environment: { HTTP_PROXY: credentialled.href, NO_PROXY: "" },
  });
  assert.equal(await (await optedIn(`${targetUrl}/fixture`)).text(), "proxy");
  assert.equal(received.length, 1);
  const expected = `Basic ${Buffer.from("operator:s3cret", "utf8").toString("base64")}`;
  assert.equal(received[0]["proxy-authorization"], expected);
  // A plain-HTTP request is forwarded verbatim, so an Authorization header built from the
  // proxy's userinfo would be handed on to the origin server as well.
  assert.equal(received[0].authorization, undefined, "proxy credentials must not become Authorization");
});

test("loopback service probes never travel through an ambient proxy", async (context) => {
  let targetHits = 0;
  let proxyHits = 0;
  const target = createServer((_request, response) => {
    targetHits += 1;
    response.end(JSON.stringify({ status: "ready" }));
  });
  const proxy = createServer((_request, response) => {
    proxyHits += 1;
    response.end("proxy");
  });
  const [targetUrl, proxyUrl] = await Promise.all([listen(target), listen(proxy)]);
  context.after(async () => Promise.all([close(target), close(proxy)]));

  const moduleUrl = new URL("./node-tools.mjs", import.meta.url).href;
  const script = [
    `const { createHttpFetch } = await import(${JSON.stringify(moduleUrl)});`,
    "const localFetch = createHttpFetch();",
    `const response = await localFetch(${JSON.stringify(`${targetUrl}/api/inference/health`)});`,
    "console.log((await response.json()).status);",
  ].join("\n");
  const result = await runNode(script, {
    ...process.env,
    HTTP_PROXY: proxyUrl,
    HTTPS_PROXY: proxyUrl,
    ALL_PROXY: proxyUrl,
    NO_PROXY: "",
    NODE_USE_ENV_PROXY: "1",
    NODE_OPTIONS: "--use-env-proxy",
  });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.trim(), "ready");
  assert.equal(targetHits, 1);
  assert.equal(proxyHits, 0, "a proxy cannot answer for a service on 127.0.0.1");

  const start = readFileSync(path.join(projectRoot, "scripts", "start.mjs"), "utf8");
  const validation = readFileSync(path.join(projectRoot, "scripts", "update-validation.mjs"), "utf8");
  for (const source of [start, validation]) {
    assert.match(source, /createHttpFetch\b/);
    assert.doesNotMatch(source, /(?<![.\w])fetch\(`http:\/\/127\.0\.0\.1/);
  }
  assert.equal(typeof createHttpFetch, "function");
});

test("the application the wizard launches keeps the proxy the installer dropped", () => {
  const gui = readFileSync(path.join(projectRoot, "scripts", "setup-gui.mjs"), "utf8");
  const snapshot = gui.indexOf("const applicationEnvironment = { ...process.env };");
  const applied = gui.indexOf("applyInstallerProxyPolicy()");
  assert.ok(snapshot >= 0, "the pristine environment must be captured");
  assert.ok(snapshot < applied, "it must be captured before the policy strips this process");
  assert.match(gui, /appProcess = spawn\(vite\.command, vite\.args, \{\s*cwd: projectRoot,\s*env: applicationEnvironment,/);

  const environment = {
    HTTPS_PROXY: "http://vpn.example:7890",
    NODE_USE_ENV_PROXY: "1",
    PATH: "test-path",
  };
  const application = { ...environment };
  applyInstallerProxyPolicy([], environment);
  assert.equal(environment.HTTPS_PROXY, undefined, "installer children lose the proxy");
  assert.equal(application.HTTPS_PROXY, "http://vpn.example:7890", "the application keeps it");
});

test("setup entry points bind HTTP and installer children to the policy", () => {
  const setup = readFileSync(path.join(projectRoot, "scripts", "setup.mjs"), "utf8");
  const gui = readFileSync(path.join(projectRoot, "scripts", "setup-gui.mjs"), "utf8");
  for (const source of [setup, gui]) {
    assert.match(source, /applyInstallerProxyPolicy\(\)/);
    assert.match(source, /createInstallerFetch\(proxyPolicy\)/);
    assert.doesNotMatch(source, /setGlobalProxyFromEnv\(\)/);
  }
  assert.match(gui, /if \(proxyPolicy\.useProxy\) throw error/);
  assert.match(setup, /if \(proxyPolicy\.useProxy\) throw error/);
  assert.match(setup, /args: \["pip", "install", "--no-config"/);
  assert.match(setup, /\["venv", "--no-config", "--seed"/);
  assert.match(setup, /UV_NO_CONFIG: "1"/);
  assert.match(setup, /npmInstallerEnvironment\(process\.env/);
  assert.match(setup, /options: \{ env: npmEnvironment \}/);
});

test("environment workflow keeps -I and injects GITHUB_WORKSPACE before backend imports", (context) => {
  const workflow = readFileSync(path.join(projectRoot, ".github", "workflows", "environment-setup.yml"), "utf8");
  const smokeCommand = workflow.match(/& \$python -I -c "([^"]+)"/)?.[1];
  assert.ok(smokeCommand, "workflow must contain the isolated backend smoke command");
  assert.match(smokeCommand, /sys\.path\.insert\(0, os\.environ\['GITHUB_WORKSPACE'\]\)/);
  assert.ok(smokeCommand.indexOf("sys.path.insert") < smokeCommand.indexOf("import backend."));

  const temporaryDirectory = mkdtempSync(path.join(os.tmpdir(), "xirai-isolated-import-"));
  context.after(() => rmSync(temporaryDirectory, { recursive: true, force: true }));
  const packageDirectory = path.join(temporaryDirectory, "workflow_fixture");
  const outsideDirectory = mkdtempSync(path.join(os.tmpdir(), "xirai-isolated-cwd-"));
  context.after(() => rmSync(outsideDirectory, { recursive: true, force: true }));
  writeFileSync(path.join(temporaryDirectory, "workflow_fixture.py"), "VALUE = 'workspace-imported'\n", "utf8");
  const pythonCommand = process.platform === "win32" ? "python" : "python3";
  const result = spawnSync(pythonCommand, [
    "-I", "-c",
    "import os,sys; sys.path.insert(0, os.environ['GITHUB_WORKSPACE']); import workflow_fixture; print(workflow_fixture.VALUE)",
  ], {
    cwd: outsideDirectory,
    env: { ...process.env, GITHUB_WORKSPACE: temporaryDirectory },
    encoding: "utf8",
    windowsHide: true,
  });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.trim(), "workspace-imported");
  assert.equal(path.basename(packageDirectory), "workflow_fixture");
});
