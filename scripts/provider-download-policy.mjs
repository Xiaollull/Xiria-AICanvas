import { fetchJsonWithDeadline, REMOTE_JSON_HARD_MAX_BYTES } from "./recommended-model-cache.mjs";

const redirectStatuses = new Set([301, 302, 303, 307, 308]);
export const PROVIDER_METADATA_JSON_MAX_BYTES = REMOTE_JSON_HARD_MAX_BYTES;

const modelOrigins = new Map([
  ["https://civitai.com", "civitai"],
  ["https://civitai.red", "civitai"],
  ["https://huggingface.co", "huggingface"],
  ["https://hf-mirror.com", "huggingface-mirror"],
  ["https://modelscope.cn", "modelscope"],
  ["https://github.com", "github"],
  ["https://githubusercontent.com", "github"],
  ["https://raw.githubusercontent.com", "github"],
  ["https://ghfast.top", "github-mirror"],
  ["https://ghproxy.net", "github-mirror"],
]);

const credentialOrigins = new Map([
  ["https://civitai.com", "civitai"],
  ["https://civitai.red", "civitai"],
  ["https://huggingface.co", "huggingface"],
  ["https://modelscope.cn", "modelscope"],
]);

const metadataOrigins = new Map([
  ["https://civitai.com", "civitai"],
  ["https://civitai.red", "civitai"],
  ["https://huggingface.co", "huggingface"],
  ["https://hf-mirror.com", "huggingface-mirror"],
  ["https://modelscope.cn", "modelscope"],
]);

function safeHttpsUrl(value, message) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(message);
  }
  if (parsed.protocol !== "https:" || parsed.username || parsed.password || parsed.port) throw new Error(message);
  return parsed;
}

export function modelDownloadProvider(value) {
  const parsed = safeHttpsUrl(value, "模型下载地址必须是受信任的 HTTPS origin");
  return modelOrigins.get(parsed.origin) || "";
}

export async function fetchProviderMetadataJson(url, {
  headers = {},
  timeoutMs = 30000,
  fetcher = globalThis.fetch,
  dispatcher,
  maximumRedirects = 5,
} = {}) {
  const parsed = safeHttpsUrl(url, "模型元数据地址必须是受信任的 HTTPS provider origin");
  const metadataProvider = metadataOrigins.get(parsed.origin);
  if (!metadataProvider) throw new Error("模型元数据地址不属于受信任的 provider origin");
  if (!Number.isSafeInteger(maximumRedirects) || maximumRedirects < 0 || maximumRedirects > 10) throw new TypeError("模型元数据重定向限制无效");
  const requestHeaders = new Headers(headers);
  let credentialProvider = "";
  if (requestHeaders.has("authorization")) {
    credentialProvider = credentialOrigins.get(parsed.origin) || "";
    if (!credentialProvider) throw new Error("拒绝向非受信任的模型 provider origin 发送凭据");
  }
  const manualRedirectFetcher = async (_url, requestOptions) => {
    let currentUrl = parsed;
    for (let redirects = 0; redirects <= maximumRedirects; redirects += 1) {
      if (requestOptions.signal?.aborted) throw Object.assign(new Error("模型元数据请求已取消"), { name: "AbortError" });
      const response = await fetcher(currentUrl.href, {
        ...requestOptions,
        headers: requestHeaders,
        redirect: "manual",
      });
      if (requestOptions.signal?.aborted) {
        await Promise.resolve(response.body?.cancel()).catch(() => {});
        throw Object.assign(new Error("模型元数据请求已取消"), { name: "AbortError" });
      }
      const location = response.headers.get("location");
      if (!location || !redirectStatuses.has(response.status)) return response;
      await Promise.resolve(response.body?.cancel()).catch(() => {});
      if (redirects === maximumRedirects) throw new Error("模型元数据重定向次数过多");
      const nextUrl = safeHttpsUrl(new URL(location, currentUrl).href, "模型元数据拒绝不安全的重定向地址");
      if (metadataOrigins.get(nextUrl.origin) !== metadataProvider) {
        throw new Error("模型元数据拒绝跨 provider 或非受信任 origin 的重定向");
      }
      if (requestHeaders.has("authorization") && credentialOrigins.get(nextUrl.origin) !== credentialProvider) {
        requestHeaders.delete("authorization");
      }
      currentUrl = nextUrl;
    }
    throw new Error("模型元数据重定向次数过多");
  };
  return fetchJsonWithDeadline(manualRedirectFetcher, parsed.href, {
    headers: requestHeaders,
    ...(dispatcher ? { dispatcher } : {}),
  }, timeoutMs, PROVIDER_METADATA_JSON_MAX_BYTES);
}

export async function fetchModelProviderDownload(url, options = {}, {
  fetcher = globalThis.fetch,
  dispatcher,
  maximumRedirects = 5,
} = {}) {
  if (typeof fetcher !== "function") throw new TypeError("model download fetcher must be a function");
  const headers = new Headers(options.headers || {});
  let currentUrl = safeHttpsUrl(url, "模型下载地址必须是受信任的 HTTPS origin");
  const initialProvider = modelOrigins.get(currentUrl.origin);
  if (!initialProvider) throw new Error("模型下载地址不属于受信任的 provider origin");
  let credentialProvider = "";
  if (headers.has("authorization")) {
    credentialProvider = credentialOrigins.get(currentUrl.origin) || "";
    if (!credentialProvider) throw new Error("拒绝向非受信任的模型 provider origin 发送凭据");
  }

  for (let redirects = 0; redirects <= maximumRedirects; redirects += 1) {
    const response = await fetcher(currentUrl.href, {
      ...options,
      headers,
      redirect: "manual",
      ...(dispatcher ? { dispatcher } : {}),
    });
    const location = response.headers.get("location");
    if (!location || !redirectStatuses.has(response.status)) return response;
    if (redirects === maximumRedirects) throw new Error("模型下载重定向次数过多");
    let nextHref;
    try {
      nextHref = new URL(location, currentUrl).href;
    } finally {
      void Promise.resolve(response.body?.cancel()).catch(() => {});
    }
    const nextUrl = safeHttpsUrl(nextHref, "模型下载拒绝不安全的重定向地址");
    if (headers.has("authorization") && credentialOrigins.get(nextUrl.origin) !== credentialProvider) {
      headers.delete("authorization");
    }
    currentUrl = nextUrl;
  }
  throw new Error("模型下载重定向次数过多");
}
