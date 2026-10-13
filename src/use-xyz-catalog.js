import { useCallback, useEffect, useRef, useState } from "react";
import { xyzCatalogChoices } from "./xyz-sidebar-context.js";

/** Read the existing engine adapters. No UI-state/cards/metadata GET, migrations, or binding reconciliation. */
export default function useXYZCatalog(engine, enabled, fetcher = globalThis.fetch) {
  const [state, setState] = useState(null), [refreshing, setRefreshing] = useState(false);
  const sequence = useRef(0), refreshController = useRef(null);
  const read = useCallback(async (signal) => {
    const request = async (url) => {
      const response = await fetcher(url, { signal });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法读取目录");
      return payload;
    };
    const [models, loras] = await Promise.all([
      request(`/api/models?engine=${encodeURIComponent(engine)}`),
      request(`/api/loras?engine=${encodeURIComponent(engine)}`),
    ]);
    if (models.engine && models.engine !== engine || loras.engine && loras.engine !== engine) throw new Error("目录引擎不匹配；保留当前草稿");
    const choices = xyzCatalogChoices({ ...models, engine });
    const directories = models.model_type === "split"
      ? Object.fromEntries(Object.keys(choices).map((field) => [field, models.assets?.[field]?.directory || ""]))
      : { checkpoint: models.directory || "" };
    return { engine, loading: false, refreshing: false, choices, directories,
      library: loras.categories || [], loraDirectory: loras.directory || "",
      loras: (loras.categories || []).flatMap((category) => (category.models || []).map((item) => ({ ...item, category: category.label }))), error: "" };
  }, [engine, fetcher]);
  useEffect(() => {
    if (!enabled) return undefined;
    const controller = new AbortController();
    const requestSequence = ++sequence.current;
    refreshController.current?.abort();
    refreshController.current = null;
    setRefreshing(false);
    setState({ engine, loading: true, refreshing: false, choices: {}, directories: {}, loras: [], error: "" });
    read(controller.signal).then((next) => {
      if (!controller.signal.aborted && sequence.current === requestSequence) setState(next);
    }).catch((error) => {
      if (!controller.signal.aborted && sequence.current === requestSequence) setState({ engine, loading: false, refreshing: false, choices: {}, directories: {}, loras: [], error: error.message });
    });
    return () => {
      controller.abort();
      refreshController.current?.abort();
      refreshController.current = null;
      sequence.current += 1;
    };
  }, [engine, enabled, read]);
  const refresh = useCallback(() => {
    if (!enabled) return Promise.resolve();
    const requestSequence = ++sequence.current;
    refreshController.current?.abort();
    const controller = new AbortController();
    refreshController.current = controller;
    setRefreshing(true);
    setState((current) => current?.engine === engine ? { ...current, refreshing: true } : current);
    return read(controller.signal).then((next) => {
      if (!controller.signal.aborted && sequence.current === requestSequence) setState(next);
    }).catch((error) => {
      if (!controller.signal.aborted && sequence.current === requestSequence) setState((current) => current?.engine === engine ? { ...current, refreshing: false, error: error.message } : current);
    }).finally(() => {
      if (sequence.current === requestSequence) {
        refreshController.current = null;
        setRefreshing(false);
      }
    });
  }, [enabled, engine, read]);
  const current = state?.engine === engine ? state : { engine, loading: enabled, refreshing: false, choices: {}, directories: {}, loras: [], error: "" };
  return { ...current, refreshing: current.refreshing || refreshing, refresh };
}
