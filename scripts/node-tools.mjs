import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

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
