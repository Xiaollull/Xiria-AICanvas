import { useCallback, useEffect, useRef, useState } from "react";

import { emptyToolboxState, normalizeToolboxState } from "./toolbox-state";

// Reads the Toolbox's saved state once and writes it back as it changes.
//
// It lives outside `ToolboxPage` so the page can hold off mounting a tool until the snapshot has
// arrived: a tool that mounts empty and is then handed its state would flash the empty view and,
// worse, would have to reconcile a restore against whatever the user had already started doing.
//
// Writes are debounced and latest-wins. Typing a directory path is one keystroke per change and
// parsing a picture rewrites the whole metadata record, so an unthrottled save would rewrite the
// file dozens of times a second.

const SAVE_DELAY = 400;

export function useToolboxState() {
  const [state, setState] = useState(null);
  const [error, setError] = useState("");
  const pendingRef = useRef(null);
  const timerRef = useRef(0);
  // Nothing may be written before the first read completes. Without this the mount-time save from
  // a tool's own initialisation would overwrite the saved state with an empty one.
  const loadedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch("/api/toolbox/state", { cache: "no-store" });
        const payload = response.ok ? await response.json().catch(() => null) : null;
        if (cancelled) return;
        setState(normalizeToolboxState(payload?.state));
      } catch {
        if (cancelled) return;
        // A Toolbox that cannot read its saved state still has to open. It starts empty and says
        // nothing, because the user did not ask for anything yet.
        setState(emptyToolboxState());
      } finally {
        if (!cancelled) loadedRef.current = true;
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const flush = useCallback(async () => {
    const next = pendingRef.current;
    if (!next) return;
    pendingRef.current = null;
    try {
      const response = await fetch("/api/toolbox/state", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state: next }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setError("");
    } catch (saveError) {
      setError(`工具箱状态未能保存：${saveError.message}`);
    }
  }, []);

  const persist = useCallback((next) => {
    if (!loadedRef.current) return;
    pendingRef.current = next;
    window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(flush, SAVE_DELAY);
  }, [flush]);

  // A save in flight when the page unmounts would be dropped, and leaving the Toolbox is exactly
  // when the state most needs to have been written.
  useEffect(() => () => {
    window.clearTimeout(timerRef.current);
    void flush();
  }, [flush]);

  const update = useCallback((change) => {
    setState((current) => {
      if (!current) return current;
      const next = normalizeToolboxState(typeof change === "function" ? change(current) : { ...current, ...change });
      persist(next);
      return next;
    });
  }, [persist]);

  const setActiveTool = useCallback((activeTool) => update((current) => ({ ...current, activeTool })), [update]);
  const setImageInfo = useCallback((imageInfo) => update((current) => ({ ...current, imageInfo })), [update]);

  return { state, ready: state !== null, error, setActiveTool, setImageInfo };
}
