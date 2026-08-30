import { createHttpFetch } from "./node-tools.mjs";

// These are process-local only. The setup launcher never writes to a user's
// shell, registry, npmrc, pip.conf, or uv configuration file.
export const proxyEnvironmentNames = [
  "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
  "http_proxy", "https_proxy", "all_proxy",
];

export const installerProxyConfigNames = [
  "NPM_CONFIG_PROXY", "NPM_CONFIG_HTTPS_PROXY",
  "NPM_CONFIG_HTTP_PROXY", "npm_config_proxy", "npm_config_https_proxy", "npm_config_http_proxy",
  "PIP_PROXY", "pip_proxy",
  "UV_HTTP_PROXY", "UV_HTTPS_PROXY", "UV_ALL_PROXY",
  "uv_http_proxy", "uv_https_proxy", "uv_all_proxy",
];

const nodeAmbientProxyNames = ["NODE_USE_ENV_PROXY"];

export function proxyOptIn(argv = process.argv.slice(2), environment = process.env) {
  return argv.includes("--use-proxy") || environment.XIRAI_USE_PROXY === "1";
}

/** Removes every spelling of `names`, not just the exact ones.
 *
 * Windows treats environment names case-insensitively and npm lowercases the whole
 * `npm_config_*` namespace, so a variable removed under one spelling and left under
 * another is still live for the child. Deleting by folded name closes that gap.
 */
function deleteVariants(environment, names) {
  const unwanted = new Set(names.map((name) => name.toLowerCase()));
  for (const name of Object.keys(environment)) {
    if (unwanted.has(name.toLowerCase())) delete environment[name];
  }
}

export function installerEnvironment(environment = process.env, { useProxy = proxyOptIn([], environment) } = {}) {
  const result = { ...environment };
  deleteVariants(result, installerProxyConfigNames);
  if (useProxy) {
    const allProxy = result.ALL_PROXY || result.all_proxy;
    if (allProxy) {
      result.HTTP_PROXY ||= allProxy;
      result.HTTPS_PROXY ||= allProxy;
    }
  } else {
    deleteVariants(result, proxyEnvironmentNames);
  }
  deleteVariants(result, nodeAmbientProxyNames);
  if (result.NODE_OPTIONS) {
    result.NODE_OPTIONS = result.NODE_OPTIONS
      .replace(/(?:^|\s)--use-env-proxy(?:=(?:true|1))?(?=\s|$)/g, " ")
      .replace(/\s+/g, " ")
      .trim();
    if (!result.NODE_OPTIONS) delete result.NODE_OPTIONS;
  }
  // NO_PROXY is deliberately neither removed nor translated into a proxy. In
  // direct mode it is inert; in opted-in mode Node and child installers retain
  // their standard bypass semantics.
  return result;
}

export function applyInstallerProxyPolicy(argv = process.argv.slice(2), environment = process.env) {
  const useProxy = proxyOptIn(argv, environment);
  const env = installerEnvironment(environment, { useProxy });
  for (const name of [...proxyEnvironmentNames, ...installerProxyConfigNames, ...nodeAmbientProxyNames, "NODE_OPTIONS"]) {
    if (name in env) environment[name] = env[name];
    else delete environment[name];
  }
  return { useProxy, configured: configuredProxy(environment), environment };
}

function configuredProxy(environment) {
  return proxyEnvironmentNames.some((name) => Boolean(environment[name]));
}

function proxyForProtocol(environment, protocol) {
  const allProxy = environment.ALL_PROXY || environment.all_proxy;
  if (protocol === "https:") {
    return environment.HTTPS_PROXY || environment.https_proxy
      || environment.HTTP_PROXY || environment.http_proxy || allProxy || null;
  }
  return environment.HTTP_PROXY || environment.http_proxy || allProxy || null;
}

function noProxyMatch(target, environment) {
  const configured = environment.NO_PROXY || environment.no_proxy;
  if (!configured) return false;
  const hostname = target.hostname.toLowerCase().replace(/^\[|\]$/g, "");
  const port = target.port || (target.protocol === "https:" ? "443" : "80");
  return configured.split(",").some((rawRule) => {
    const rule = rawRule.trim().toLowerCase();
    if (!rule) return false;
    if (rule === "*") return true;
    const bracketed = rule.match(/^\[([^\]]+)](?::(\d+))?$/);
    const plain = !bracketed && rule.match(/^([^:]+)(?::(\d+))?$/);
    const ruleHost = (bracketed?.[1] || plain?.[1] || "").replace(/^\./, "");
    const rulePort = bracketed?.[2] || plain?.[2];
    if (!ruleHost || rulePort && rulePort !== port) return false;
    return hostname === ruleHost || hostname.endsWith(`.${ruleHost}`);
  });
}

/** npm receives a project-owned user config and process-high proxy values.
 * `null` is npm's explicit no-proxy value, and outranks any global npmrc.
 */
export function npmInstallerEnvironment(environment, { useProxy, userConfig }) {
  const result = installerEnvironment(environment, { useProxy });
  // Every `npm run` exports `npm_config_userconfig` pointing at the user's own npmrc, and
  // npm folds that name together with `NPM_CONFIG_USERCONFIG` and lets whichever it reads
  // last win. Adding the project-owned spelling therefore only shadows the inherited one by
  // insertion order; the inherited name has to go for the boundary to actually hold.
  deleteVariants(result, ["NPM_CONFIG_USERCONFIG"]);
  result.NPM_CONFIG_USERCONFIG = userConfig;
  result.NPM_CONFIG_PROXY = useProxy ? proxyForProtocol(result, "http:") || "null" : "null";
  result.NPM_CONFIG_HTTPS_PROXY = useProxy ? proxyForProtocol(result, "https:") || "null" : "null";
  return result;
}

/** A fetch function bound to the installer policy, never to a later ambient-env read. */
export function createInstallerFetch({ useProxy, environment = process.env } = {}) {
  const hasProxy = useProxy && configuredProxy(environment);
  if (useProxy && !hasProxy) {
    return async () => { throw new Error("已显式启用代理，但未设置 HTTP_PROXY、HTTPS_PROXY 或 ALL_PROXY"); };
  }
  const policyFetch = createHttpFetch({
    resolveProxy: (target) => hasProxy && !noProxyMatch(target, environment)
      ? proxyForProtocol(environment, target.protocol)
      : null,
  });

  return async (url, options = {}) => {
    try {
      return await policyFetch(url, options);
    } catch (error) {
      if (hasProxy && !String(error.message).includes("已显式启用代理")) {
        throw new Error(`已显式启用代理，但请求失败：${error.message}`, { cause: error });
      }
      throw error;
    }
  };
}
