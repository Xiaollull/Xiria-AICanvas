// Browser-safe owner predicate shared by Start and the configurator return path.
export const APPLICATION_PROTOCOL = 40;
export function isApplicationReady(health, workspaceId) {
  return typeof workspaceId === "string" && workspaceId.length > 0 &&
    health?.status === "ready" && health.protocol === APPLICATION_PROTOCOL && health.workspace_id === workspaceId;
}
