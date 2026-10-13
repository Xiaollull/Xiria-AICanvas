const network = window.fetch.bind(window);
await import("./xyz-full-ui-stage2.fixture.jsx");
const fixture = window.__appFetch;
window.__taskCalls = [];
window.__appFetch = async (url, method, body) => {
  if (url.startsWith("/api/inference/xyz-plots") || url.startsWith("/api/inference/lora-mix")) {
    window.__taskCalls.push({ url, method, body });
    return network(url, { method, headers: { "Content-Type": "application/json" }, ...(body ? { body: JSON.stringify(body) } : {}) });
  }
  return fixture(url, method, body);
};
