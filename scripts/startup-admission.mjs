import net from "node:net";
import { isApplicationReady } from "./service-readiness.js";

export function startupDisposition(health, workspaceId) {
  if (isApplicationReady(health, workspaceId)) return "reuse";
  return health == null ? "probe-ports" : "blocked";
}
export async function assertStartupPortsFree(ports, createServer = net.createServer) {
  for (const { port, host } of ports) {
    await new Promise((resolve, reject) => {
      const server = createServer();
      server.once("error", () => reject(new Error(`端口 ${port} 已占用或不可用；未停止任何服务。请核对已有服务后手动处理。`)));
      server.listen({ port, host, exclusive: true }, () => server.close((error) => error ? reject(error) : resolve()));
    });
  }
}
