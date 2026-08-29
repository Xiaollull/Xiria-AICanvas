const terminalItemStatuses = new Set(["complete", "error"]);

function normalizedDigest(value) {
  return typeof value === "string" && /^[a-f0-9]{64}$/i.test(value) ? value.toLowerCase() : "";
}

export function filterPendingRecommendedArtifacts(artifacts, installed, queueItems = []) {
  const installedDigests = installed instanceof Map ? new Set(installed.keys()) : new Set(installed || []);
  const queuedDigests = new Set(queueItems
    .filter((item) => item.status !== "complete")
    .map((item) => normalizedDigest(item.sha256))
    .filter(Boolean));
  return artifacts.filter((artifact) => {
    const digest = normalizedDigest(artifact.sha256);
    return !digest || (!installedDigests.has(digest) && !queuedDigests.has(digest));
  });
}

export function appendDownloadQueueState(state, { source, kind, engine = "", items }) {
  const startIndex = (state.items || []).length + 1;
  const appendedItems = items.map((item, offset) => ({
    index: startIndex + offset,
    url: String(item.label || item.url || "").slice(0, 240),
    status: "waiting",
    source,
    kind,
    engine,
    ...(item.id ? { artifactId: item.id } : {}),
    ...(normalizedDigest(item.sha256) ? { sha256: normalizedDigest(item.sha256) } : {}),
  }));
  const targetKey = `${kind}\u0000${engine}`;
  const targets = [...(state.targets || [])];
  if (!targets.some((target) => `${target.kind}\u0000${target.engine || ""}` === targetKey)) targets.push({ kind, engine });
  return {
    startIndex,
    state: {
      ...state,
      totalModels: (state.totalModels || 0) + appendedItems.length,
      items: [...(state.items || []), ...appendedItems],
      targets,
    },
  };
}

export function itemStatusIsTerminal(status) {
  return terminalItemStatuses.has(status);
}
