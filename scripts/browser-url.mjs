import { spawn } from "node:child_process";

// Like ComfyUI's webbrowser.open: one HTTP URL goes to the OS default handler.
// Reuse/window choice belongs to that handler, never to profile selection here.
export function browserUrlCommand(url, platform = process.platform) {
  const target = new URL(url);
  if (!["http:", "https:"].includes(target.protocol) || target.username || target.password || /[\r\n"&|<>^]/.test(url)) {
    throw new Error("浏览器入口必须是安全的 HTTP(S) 地址");
  }
  if (platform === "win32") return { command: "cmd.exe", args: ["/d", "/s", "/c", "start", "", url] };
  return { command: platform === "darwin" ? "open" : "xdg-open", args: [url] };
}
export function openBrowser(url, { platform = process.platform, spawnProcess = spawn, report = console.log } = {}) {
  const { command, args } = browserUrlCommand(url, platform);
  const child = spawnProcess(command, args, { detached: true, stdio: "ignore", windowsHide: true });
  child.on("error", () => report(`请在浏览器打开：${url}`));
  child.unref();
  return child;
}
