import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  fetchModelProviderDownload,
  fetchProviderMetadataJson,
  PROVIDER_METADATA_JSON_MAX_BYTES,
} from "./provider-download-policy.mjs";

function authorization(options) {
  return new Headers(options.headers || {}).get("authorization");
}

function response(status = 200, location = "") {
  return {
    status,
    headers: new Headers(location ? { Location: location } : {}),
    body: { cancel: async () => {} },
  };
}

test("Civitai, Hugging Face, and ModelScope credentials reach each legal first request", async () => {
  for (const url of [
    "https://civitai.com/api/download/models/12",
    "https://civitai.red/api/download/models/12",
    "https://huggingface.co/o/r/resolve/main/model.safetensors",
    "https://modelscope.cn/models/o/r/resolve/main/model.safetensors",
  ]) {
    const calls = [];
    const result = await fetchModelProviderDownload(url, { headers: { Authorization: "Bearer provider-secret" } }, {
      fetcher: async (target, options) => {
        calls.push({ target, authorization: authorization(options), redirect: options.redirect });
        return response(206);
      },
    });
    assert.equal(result.status, 206);
    assert.deepEqual(calls, [{ target: url, authorization: "Bearer provider-secret", redirect: "manual" }]);
  }
});

test("provider credentials are rejected before fetch for HTTP, foreign, mirror, credential, and nonstandard-port origins", async () => {
  let calls = 0;
  const fetcher = async () => { calls += 1; return response(); };
  for (const url of [
    "http://civitai.com/api/download/models/12",
    "https://evil.example/model",
    "https://hf-mirror.com/o/r/model",
    "https://user:pass@huggingface.co/o/r/model",
    "https://modelscope.cn:444/o/r/model",
  ]) {
    await assert.rejects(fetchModelProviderDownload(url, { headers: { Authorization: "Bearer provider-secret" } }, { fetcher }), /HTTPS origin|受信任/);
  }
  assert.equal(calls, 0);
});

test("cross-provider and object-storage redirects strip Authorization permanently", async () => {
  const calls = [];
  const responses = [
    response(302, "https://objects.example/signed/model"),
    response(302, "https://civitai.red/api/download/models/12"),
    response(206),
  ];
  await fetchModelProviderDownload("https://civitai.com/api/download/models/12", { headers: { Authorization: "Bearer civitai-secret" } }, {
    fetcher: async (target, options) => {
      calls.push({ target, authorization: authorization(options) });
      return responses.shift();
    },
  });
  assert.deepEqual(calls, [
    { target: "https://civitai.com/api/download/models/12", authorization: "Bearer civitai-secret" },
    { target: "https://objects.example/signed/model", authorization: null },
    { target: "https://civitai.red/api/download/models/12", authorization: null },
  ]);
});

test("Authorization survives only redirects to an explicitly allowed origin of the same provider", async () => {
  const civitaiCalls = [];
  await fetchModelProviderDownload("https://civitai.com/api/download/models/12", { headers: { Authorization: "Bearer civitai-secret" } }, {
    fetcher: async (target, options) => {
      civitaiCalls.push({ target, authorization: authorization(options) });
      return civitaiCalls.length === 1 ? response(307, "https://civitai.red/api/download/models/12") : response(206);
    },
  });
  assert.deepEqual(civitaiCalls.map((call) => call.authorization), ["Bearer civitai-secret", "Bearer civitai-secret"]);

  const huggingfaceCalls = [];
  await fetchModelProviderDownload("https://huggingface.co/o/r/resolve/main/model", { headers: { Authorization: "Bearer hf-secret" } }, {
    fetcher: async (target, options) => {
      huggingfaceCalls.push({ target, authorization: authorization(options) });
      return huggingfaceCalls.length === 1 ? response(302, "https://cdn-lfs.hf.co/signed/model") : response(206);
    },
  });
  assert.deepEqual(huggingfaceCalls.map((call) => call.authorization), ["Bearer hf-secret", null]);
});

test("redirects to non-HTTPS destinations are rejected after cancelling the source response", async () => {
  let cancelled = false;
  await assert.rejects(fetchModelProviderDownload("https://modelscope.cn/models/o/r/model", { headers: { Authorization: "Bearer ms-secret" } }, {
    fetcher: async () => ({
      status: 302,
      headers: new Headers({ Location: "http://modelscope.cn/insecure" }),
      body: { cancel: async () => { cancelled = true; } },
    }),
  }), /不安全的重定向地址/);
  assert.equal(cancelled, true);
});

test("Vite wires model downloads to provider policy and keeps update archives on GitHub policy", async () => {
  const source = await readFile(new URL("../vite.config.js", import.meta.url), "utf8");
  assert.equal((source.match(/fetcher: fetchProviderDownload/g) || []).length, 4);
  assert.match(source, /fetcher: \(downloadUrl, options\) => fetchUpdateDownload\(downloadUrl, options/);
  assert.match(source, /onlineUpdateNetworkInternals = \{ fetchBoundedUpdateBody, fetchDownload: fetchUpdateDownload \}/);
  const providerJson = source.slice(source.indexOf("async function fetchProviderJson"), source.indexOf("function selectDownloadableFile"));
  assert.match(providerJson, /const result = await fetchProviderMetadataJson\(url/);
  assert.doesNotMatch(providerJson, /response\.json\(\)/);
  assert.doesNotMatch(source, /fetchJsonWithDeadline/);
  const civitaiResolution = source.slice(source.indexOf("async function resolveCivitaiDownload"), source.indexOf("function parseModelScopeLocation"));
  assert.match(civitaiResolution, /const headers = providerHeaders\(apiKey\)/);
  assert.match(civitaiResolution, /fetchProviderJson\(target, headers, metadataTimeout\)/);
  const recommendedFamily = source.slice(source.indexOf("async function fetchCivitaiRecommendedFamily"), source.indexOf("async function fetchRecommendedRemoteSnapshot"));
  assert.match(recommendedFamily, /raceTrustedCivitaiFamilyVersions/);
  assert.match(recommendedFamily, /fetchProviderJson\(`https:\/\/\$\{domain\}\/api\/v1\/models\/\$\{family\.modelId\}`, \{\}, recommendedCivitaiTimeout\)/);
});

test("provider metadata deadline remains armed while the response body hangs", async () => {
  let requestSignal;
  await assert.rejects(fetchProviderMetadataJson("https://civitai.com/api/v1/models/34", {
    timeoutMs: 15,
    fetcher: async (_url, options) => {
      requestSignal = options.signal;
      return {
        headers: { get: () => null },
        body: { getReader: () => ({ read: () => new Promise(() => {}), releaseLock() {} }) },
      };
    },
  }), (error) => error.code === "ETIMEDOUT");
  assert.equal(requestSignal.aborted, true);
});

test("provider metadata rejects declared and streamed bodies above four MiB", async () => {
  let declaredBodyRead = false;
  await assert.rejects(fetchProviderMetadataJson("https://huggingface.co/api/models/o/r", {
    fetcher: async () => ({
      headers: { get: () => String(PROVIDER_METADATA_JSON_MAX_BYTES + 1) },
      body: { getReader: () => ({ read: async () => { declaredBodyRead = true; return { done: true }; }, releaseLock() {} }) },
    }),
  }), (error) => error.code === "ERESPONSESIZE");
  assert.equal(declaredBodyRead, false);

  let sent = false;
  await assert.rejects(fetchProviderMetadataJson("https://modelscope.cn/api/v1/models/o/r", {
    fetcher: async () => ({
      headers: { get: () => null },
      body: {
        getReader: () => ({
          read: async () => {
            if (sent) return { done: true };
            sent = true;
            return { done: false, value: new Uint8Array(PROVIDER_METADATA_JSON_MAX_BYTES + 1) };
          },
          releaseLock() {},
        }),
      },
    }),
  }), (error) => error.code === "ERESPONSESIZE");
});

test("provider metadata rejects external, HTTP, localhost, private, and credential redirects before access", async () => {
  const targets = [
    "https://evil.example/metadata",
    "https://huggingface.co/api/models/o/r",
    "http://civitai.red/api/v1/models/34",
    "https://localhost/api/v1/models/34",
    "https://127.0.0.1/api/v1/models/34",
    "https://192.168.1.20/api/v1/models/34",
    "https://user:pass@civitai.red/api/v1/models/34",
  ];
  for (const target of targets) {
    const calls = [];
    let cancelled = 0;
    await assert.rejects(fetchProviderMetadataJson("https://civitai.com/api/v1/models/34", {
      headers: { Authorization: "Bearer civitai-secret" },
      fetcher: async (url, options) => {
        calls.push({ url, authorization: authorization(options) });
        if (calls.length > 1) throw new Error("rejected redirect target was accessed");
        return {
          status: 302,
          headers: new Headers({ Location: target }),
          body: { cancel: async () => { cancelled += 1; } },
        };
      },
    }), /不安全|跨 provider|非受信任/);
    assert.deepEqual(calls, [{ url: "https://civitai.com/api/v1/models/34", authorization: "Bearer civitai-secret" }]);
    assert.equal(cancelled, 1);
  }
});

test("provider metadata limits redirects and cancels every intermediate response", async () => {
  const calls = [];
  let cancelled = 0;
  await assert.rejects(fetchProviderMetadataJson("https://huggingface.co/api/models/o/r", {
    maximumRedirects: 2,
    fetcher: async (url, options) => {
      calls.push({ url, redirect: options.redirect });
      return {
        status: 307,
        headers: new Headers({ Location: `/api/models/o/r?hop=${calls.length}` }),
        body: { cancel: async () => { cancelled += 1; } },
      };
    },
  }), /重定向次数过多/);
  assert.equal(calls.length, 3);
  assert.equal(calls.every((call) => call.redirect === "manual"), true);
  assert.equal(cancelled, 3);
});

test("provider metadata follows an allowed same-provider redirect under one signal and preserves its token", async () => {
  const calls = [];
  let cancelled = 0;
  const result = await fetchProviderMetadataJson("https://civitai.com/api/v1/models/34", {
    headers: { Authorization: "Bearer civitai-secret" },
    fetcher: async (url, options) => {
      calls.push({ url, authorization: authorization(options), redirect: options.redirect, signal: options.signal });
      if (calls.length === 1) {
        return {
          status: 302,
          headers: new Headers({ Location: "https://civitai.red/api/v1/models/34" }),
          body: { cancel: async () => { cancelled += 1; } },
        };
      }
      return new Response('{"id":34,"modelVersions":[]}', { status: 200 });
    },
  });
  assert.deepEqual(result.body, { id: 34, modelVersions: [] });
  assert.deepEqual(calls.map(({ url, authorization: value, redirect }) => ({ url, authorization: value, redirect })), [
    { url: "https://civitai.com/api/v1/models/34", authorization: "Bearer civitai-secret", redirect: "manual" },
    { url: "https://civitai.red/api/v1/models/34", authorization: "Bearer civitai-secret", redirect: "manual" },
  ]);
  assert.equal(calls[0].signal, calls[1].signal);
  assert.equal(cancelled, 1);
});
