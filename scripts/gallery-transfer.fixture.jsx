// Gallery import/export browser fixture: the real App with the usual in-page stand-ins, except that every
// gallery request (JSON, image, ZIP upload and download) goes over real HTTP to the private Python fixture.
// The stand-in fetch parses JSON bodies, so it must never see a ZIP upload; gallery calls bypass it entirely.
const network = window.fetch.bind(window);
await import("./xyz-sol-app.fixture.jsx");
const fixture = window.fetch;
window.__galleryCalls = [];
window.fetch = (input, options = {}) => {
  const url = typeof input === "string" ? input : input.url;
  if (url.startsWith("/api/inference/gallery")) {
    window.__galleryCalls.push({ url, method: String(options.method || "GET").toUpperCase() });
    return network(input, options);
  }
  return fixture(input, options);
};
