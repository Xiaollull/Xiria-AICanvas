// Every path the user types into this app names a folder on the machine running
// the control plane — not on the machine showing the browser. Those are often
// the same computer, but the WebUI binds `0.0.0.0` and prints LAN URLs, so a
// phone or a second desktop can be the one asking. That makes `navigator` the
// wrong source for a path hint: it would offer `C:\...` to someone whose models
// live on a Linux box. Vite substitutes the host's own platform at build and
// dev-server start instead, which is the machine the paths actually refer to.
//
// The `typeof` guard is what keeps this importable outside a Vite build (tests
// load it directly under Node), where the constant was never substituted.
export const HOST_PLATFORM = typeof __XIRAI_HOST_PLATFORM__ === "string" ? __XIRAI_HOST_PLATFORM__ : "";

export function hostIsWindows(platform = HOST_PLATFORM) {
  return platform === "win32";
}

// Examples, not validation: the server decides what it will accept. These only
// have to show the shape of an absolute path on the host so the user recognises
// what is being asked for.
export const HOST_PATH_EXAMPLES = {
  modelsRoot: { win32: "F:\\ComfyUI\\models", posix: "/home/you/ComfyUI/models" },
  loraDirectory: { win32: "D:\\Stable-diffusion-webui\\models\\Lora", posix: "/home/you/stable-diffusion-webui/models/Lora" },
  imageDirectory: { win32: "D:\\outputs\\2026-08", posix: "/home/you/outputs/2026-08" },
};

export function hostPathExample(kind, platform = HOST_PLATFORM) {
  const example = HOST_PATH_EXAMPLES[kind];
  if (!example) return "";
  return hostIsWindows(platform) ? example.win32 : example.posix;
}
