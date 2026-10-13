// Fresh headless Chromium profile only. Mock the tagger response because no
// weights are installed in this isolated backend; Python inference is tested
// separately against the read-only F: model.
import { spawn } from "node:child_process";
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { createServer as netServer } from "node:net";

const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
async function port() {
  const server = netServer();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const value = server.address().port;
  await new Promise((resolve) => server.close(resolve));
  return value;
}
async function wait(check, timeout = 30000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    try { const value = await check(); if (value) return value; } catch { /* loading */ }
    await new Promise((resolve) => setTimeout(resolve, 180));
  }
  throw new Error("browser wait timed out");
}

export async function browserSmoke({ origin, scratch, evidenceDirectory }) {
  const debugPort = await port();
  const browser = spawn(EDGE, ["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
    "--disable-extensions", `--remote-debugging-port=${debugPort}`, `--user-data-dir=${path.join(scratch, "chromium-profile")}`,
    "--window-size=1280,800", "about:blank"], { stdio: "ignore", windowsHide: true });
  let ws;
  try {
    const tab = await wait(async () => {
      const response = await fetch(`http://127.0.0.1:${debugPort}/json/list`);
      return (await response.json()).find((entry) => entry.type === "page" && entry.webSocketDebuggerUrl);
    });
    ws = new WebSocket(tab.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => { ws.addEventListener("open", resolve, { once: true }); ws.addEventListener("error", reject, { once: true }); });
    let serial = 0;
    const pending = new Map();
    function send(method, params = {}) {
      const id = ++serial;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        ws.send(JSON.stringify({ id, method, params }));
      });
    }
    ws.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.id) {
        const promise = pending.get(message.id);
        pending.delete(message.id);
        if (message.error) promise?.reject(new Error(message.error.message)); else promise?.resolve(message.result);
      }
      if (message.method === "Fetch.requestPaused") {
        const url = message.params.request.url;
        const catalog = { models: [{ id: "wd-swinv2-v3", installed: true }], providers: ["CPUExecutionProvider"],
          default: "wd-swinv2-v3", batch_root: path.join(scratch, "outputs") };
        const tags = { rating: { general: .72, sensitive: .11, questionable: .12, explicit: .05 },
          tags: [{ name: "test tag", confidence: .91 }] };
        const payload = url.includes("/wd14/models") ? catalog : tags;
        void send("Fetch.fulfillRequest", { requestId: message.params.requestId, responseCode: 200,
          responseHeaders: [{ name: "Content-Type", value: "application/json" },
            { name: "Access-Control-Allow-Origin", value: "*" }],
          body: Buffer.from(JSON.stringify(payload)).toString("base64") }).catch(() => {});
      }
    });
    await send("Page.enable");
    await send("Runtime.enable");
    await send("DOM.enable");
    await send("Fetch.enable", { patterns: [
      { urlPattern: "*/api/inference/wd14/models*" }, { urlPattern: "*/api/inference/wd14/tag*" },
    ] });
    await send("Page.navigate", { url: origin });
    const evaluate = async (expression) => {
      const result = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
      if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
      return result.result.value;
    };
    await wait(() => evaluate("Boolean(document.querySelector('.main-nav'))"));
    const order = await evaluate("[...document.querySelectorAll('.main-nav button')].map(x=>x.textContent.trim())");
    if (order.indexOf("图生图") + 1 !== order.indexOf("图片反推") || order.indexOf("图片反推") + 1 !== order.indexOf("画廊")) {
      throw new Error("tagger is not between image-to-image and gallery");
    }
    await evaluate("[...document.querySelectorAll('.main-nav button')].find(x=>x.textContent.includes('图片反推')).click()");
    await wait(() => evaluate("Boolean(document.querySelector('.wd14-page select'))"));
    const picture = path.join(scratch, "synthetic.png");
    await writeFile(picture, Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/pXcAAAAASUVORK5CYII=", "base64"));
    const dom = await send("DOM.getDocument");
    const input = await send("DOM.querySelector", { nodeId: dom.root.nodeId, selector: ".wd14-page input[type=file]" });
    await send("DOM.setFileInputFiles", { nodeId: input.nodeId, files: [picture] });
    // Single-file selection now replaces the input and automatically starts tagging.
    await wait(() => evaluate("document.querySelector('.wd14-results textarea')?.value === 'test tag' && document.querySelectorAll('.wd14-history-strip button').length === 1"));
    await wait(() => evaluate("!document.querySelector('.app-loader')"), 45000);
    const desktop = await evaluate("({nav:document.querySelector('.main-nav button.active')?.textContent.trim(),tag:document.querySelector('.wd14-results textarea')?.value,ratings:document.querySelectorAll('.wd14-ratings span').length})");
    console.log("browser desktop", JSON.stringify(desktop));
    const stamp = Date.now();
    const screenshot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
    const desktopPath = path.join(evidenceDirectory, `wd14-desktop-${stamp}.png`);
    await writeFile(desktopPath, Buffer.from(screenshot.data, "base64"));
    await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
    await wait(() => evaluate("window.innerWidth === 390"));
    const mobile = await evaluate("({width:innerWidth,page:document.querySelector('.wd14-page')?.getBoundingClientRect().width,scrollWidth:document.documentElement.scrollWidth,nav:document.querySelector('.main-nav button.active')?.textContent.trim(),tag:document.querySelector('.wd14-results textarea')?.value})");
    console.log("browser 390px", JSON.stringify(mobile));
    const mobileImage = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
    const mobilePath = path.join(evidenceDirectory, `wd14-mobile-${stamp}.png`);
    await writeFile(mobilePath, Buffer.from(mobileImage.data, "base64"));
    await evaluate("document.querySelector('.wd14-results').scrollIntoView({block:'center',behavior:'instant'})");
    const mobileResults = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
    const mobileResultPath = path.join(evidenceDirectory, `wd14-mobile-result-${stamp}.png`);
    await writeFile(mobileResultPath, Buffer.from(mobileResults.data, "base64"));
    console.log("browser screenshots", desktopPath, mobilePath, mobileResultPath);
  } finally {
    ws?.close();
    browser.kill();
    await new Promise((resolve) => { if (browser.exitCode !== null) resolve(); else { browser.once("exit", resolve); setTimeout(resolve, 5000); } });
  }
}
