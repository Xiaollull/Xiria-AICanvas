export function normalizedDownloadCredentials({ civitai = "", huggingface = "", modelscope = "" } = {}) {
  return {
    civitai: typeof civitai === "string" ? civitai.trim() : "",
    huggingface: typeof huggingface === "string" ? huggingface.trim() : "",
    modelscope: typeof modelscope === "string" ? modelscope.trim() : "",
  };
}

export function configuredDownloadCredentialCount(credentials) {
  return Object.values(normalizedDownloadCredentials(credentials)).filter(Boolean).length;
}

function selectedOrFirst(items, selectedId) {
  if (!Array.isArray(items) || !items.length) return "";
  return items.some((item) => item?.id === selectedId) ? selectedId : items[0]?.id || "";
}

export function reconcileRecommendedSelection(families, familyId, selection = {}) {
  const family = Array.isArray(families) ? families.find((item) => item?.id === familyId) : null;
  if (!family) return { familyId: "", selection: { modelId: "", textEncoderId: "", vaeId: "" } };
  return {
    familyId,
    selection: {
      modelId: selectedOrFirst(family.models, selection.modelId),
      textEncoderId: selectedOrFirst(family.textEncoders, selection.textEncoderId),
      vaeId: selectedOrFirst(family.vaes, selection.vaeId),
    },
  };
}

export function createLatestRequestGate() {
  let generation = 0;
  return Object.freeze({
    begin() {
      generation += 1;
      return generation;
    },
    invalidate() {
      generation += 1;
      return generation;
    },
    isCurrent(candidate) {
      return candidate === generation;
    },
    commit(candidate, callback) {
      if (candidate !== generation) return false;
      callback();
      return true;
    },
  });
}
