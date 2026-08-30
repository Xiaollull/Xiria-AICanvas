import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import * as http from "node:http";
import * as https from "node:https";
import net from "node:net";
import path from "node:path";
import { Readable } from "node:stream";
import tls from "node:tls";

export function npmInvocation() {
  if (process.env.npm_execpath && existsSync(process.env.npm_execpath)) {
    return { command: process.execPath, args: [process.env.npm_execpath] };
  }
  if (process.platform === "win32") {
    const located = spawnSync("where.exe", ["npm.cmd"], { encoding: "utf8", windowsHide: true });
    const npmCommand = located.status === 0 ? located.stdout.split(/\r?\n/).find(Boolean)?.trim() : null;
    if (npmCommand) {
      const npmCli = path.join(path.dirname(npmCommand), "node_modules", "npm", "bin", "npm-cli.js");
      if (existsSync(npmCli)) return { command: process.execPath, args: [npmCli] };
    }
    throw new Error("Node.js 已安装，但找不到 npm CLI");
  }
  return { command: "npm", args: [] };
}

export function viteInvocation(projectRoot, args = []) {
  const viteEntry = path.join(projectRoot, "node_modules", "vite", "bin", "vite.js");
  if (!existsSync(viteEntry)) throw new Error("Vite 尚未安装，请先完成环境配置");
  return { command: process.execPath, args: [viteEntry, ...args] };
}

function requestHeaders(value) {
  return Object.fromEntries(new Headers(value || {}).entries());
}

function responseHeaders(rawHeaders) {
  const headers = new Headers();
  for (const [name, value] of Object.entries(rawHeaders)) {
    if (Array.isArray(value)) value.forEach((item) => headers.append(name, item));
    else if (value != null) headers.set(name, value);
  }
  return headers;
}

function proxyAuthorization(proxy) {
  if (!proxy.username && !proxy.password) return null;
  const credentials = `${decodeURIComponent(proxy.username)}:${decodeURIComponent(proxy.password)}`;
  return `Basic ${Buffer.from(credentials, "utf8").toString("base64")}`;
}

function targetAuthority(target) {
  const defaultPort = target.protocol === "https:" ? "443" : "80";
  return `${target.hostname}:${target.port || defaultPort}`;
}

function proxyTransport(proxy) {
  if (!new Set(["http:", "https:"]).has(proxy.protocol)) {
    throw new Error(`仅支持 HTTP(S) 代理，当前代理协议为 ${proxy.protocol}`);
  }
  return proxy.protocol === "https:" ? https : http;
}

function tunnelAgent(target, proxy, signal) {
  const agent = new https.Agent({ keepAlive: false });
  agent.createConnection = (_options, callback) => {
    let settled = false;
    const finish = (error, socket) => {
      if (settled) return;
      settled = true;
      callback(error, socket);
    };
    const authorization = proxyAuthorization(proxy);
    const headers = { Host: targetAuthority(target) };
    if (authorization) headers["Proxy-Authorization"] = authorization;
    const request = proxyTransport(proxy).request(proxy, {
      method: "CONNECT",
      path: targetAuthority(target),
      headers,
      // Userinfo on the proxy URL is Proxy-Authorization material and nothing else.
      // Node turns a surviving `auth` into an Authorization header of its own, so it
      // is cleared here and carried only by the header built above.
      auth: undefined,
      agent: proxy.protocol === "https:" ? new https.Agent({ keepAlive: false }) : new http.Agent({ keepAlive: false }),
      signal,
    });
    request.once("connect", (response, socket, head) => {
      if (response.statusCode !== 200) {
        socket.destroy();
        finish(new Error(`代理 CONNECT 失败（HTTP ${response.statusCode || "未知"}）`));
        return;
      }
      if (head.length) socket.unshift(head);
      const secureSocket = tls.connect({
        socket,
        servername: net.isIP(target.hostname) ? undefined : target.hostname,
        ALPNProtocols: ["http/1.1"],
      });
      secureSocket.once("secureConnect", () => finish(null, secureSocket));
      secureSocket.once("error", (error) => finish(error));
    });
    request.once("error", (error) => finish(error));
    request.end();
    return undefined;
  };
  return agent;
}

function sendBody(request, body) {
  if (body == null) {
    request.end();
    return;
  }
  if (typeof body === "string" || Buffer.isBuffer(body) || ArrayBuffer.isView(body)) {
    request.end(body);
    return;
  }
  throw new TypeError("安装器 HTTP adapter 仅支持字符串或二进制请求体");
}

/**
 * Fetch-compatible HTTP(S) adapter with an explicit per-request route.
 *
 * It always supplies a private core HTTP(S) agent. Consequently neither
 * global fetch nor a process-global agent changed by --use-env-proxy can
 * influence a direct request. A proxy route is likewise attached only to the
 * request being made; this module never changes Node's global dispatcher.
 */
export function createHttpFetch({ resolveProxy = () => null, maxRedirects = 10 } = {}) {
  const directAgents = {
    "http:": new http.Agent({ keepAlive: false }),
    "https:": new https.Agent({ keepAlive: false }),
  };

  async function requestUrl(input, options = {}, redirects = 0) {
    if (redirects > maxRedirects) throw new Error("下载重定向次数过多");
    const target = new URL(input);
    if (!new Set(["http:", "https:"]).has(target.protocol)) throw new Error(`不支持的下载协议：${target.protocol}`);
    const proxyValue = resolveProxy(target);
    const proxy = proxyValue ? new URL(proxyValue) : null;
    const headers = requestHeaders(options.headers);
    const method = String(options.method || "GET").toUpperCase();

    return new Promise((resolve, reject) => {
      let client;
      let requestTarget;
      let requestOptions;
      try {
        if (!proxy) {
          client = target.protocol === "https:" ? https : http;
          requestTarget = target;
          requestOptions = { method, headers, agent: directAgents[target.protocol], signal: options.signal };
        } else if (target.protocol === "http:") {
          client = proxyTransport(proxy);
          requestTarget = proxy;
          headers.host = target.host;
          const authorization = proxyAuthorization(proxy);
          if (authorization) headers["proxy-authorization"] = authorization;
          requestOptions = {
            method,
            path: target.href,
            headers,
            // A plain-HTTP request is forwarded verbatim, so an Authorization header
            // synthesised from the proxy's userinfo would reach the origin server too.
            // Proxy credentials travel in proxy-authorization above and nowhere else.
            auth: undefined,
            agent: proxy.protocol === "https:" ? new https.Agent({ keepAlive: false }) : new http.Agent({ keepAlive: false }),
            signal: options.signal,
          };
        } else {
          client = https;
          requestTarget = target;
          requestOptions = { method, headers, agent: tunnelAgent(target, proxy, options.signal), signal: options.signal };
        }
      } catch (error) {
        reject(error);
        return;
      }

      const request = client.request(requestTarget, requestOptions, (response) => {
        const location = response.headers.location;
        const redirectStatus = new Set([301, 302, 303, 307, 308]).has(response.statusCode);
        if (location && redirectStatus) {
          if (options.redirect === "error") {
            response.resume();
            reject(new Error(`下载返回重定向（HTTP ${response.statusCode}）`));
            return;
          }
          if (options.redirect !== "manual") {
            response.resume();
            const nextTarget = new URL(location, target);
            const nextHeaders = new Headers(headers);
            nextHeaders.delete("host");
            nextHeaders.delete("proxy-authorization");
            if (nextTarget.origin !== target.origin) nextHeaders.delete("authorization");
            const switchToGet = response.statusCode === 303 && method !== "HEAD"
              || [301, 302].includes(response.statusCode) && method === "POST";
            if (switchToGet) {
              nextHeaders.delete("content-length");
              nextHeaders.delete("content-type");
            }
            requestUrl(nextTarget, {
              ...options,
              method: switchToGet ? "GET" : method,
              headers: nextHeaders,
              body: switchToGet ? undefined : options.body,
            }, redirects + 1).then(resolve, reject);
            return;
          }
        }
        if (proxy && response.statusCode === 407) {
          response.resume();
          reject(new Error("代理认证失败（HTTP 407）"));
          return;
        }
        const bodyAllowed = method !== "HEAD" && ![204, 205, 304].includes(response.statusCode);
        if (!bodyAllowed) response.resume();
        resolve(new Response(bodyAllowed ? Readable.toWeb(response) : null, {
          status: response.statusCode,
          statusText: response.statusMessage,
          headers: responseHeaders(response.headers),
        }));
      });
      request.once("error", reject);
      try {
        sendBody(request, options.body);
      } catch (error) {
        request.destroy(error);
      }
    });
  }

  return requestUrl;
}
