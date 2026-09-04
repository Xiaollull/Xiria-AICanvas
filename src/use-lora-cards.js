import { useCallback, useEffect, useRef, useState } from "react";

import { LORA_SYNC_CHANNEL } from "./lora-state.js";
import {
  emptyLoraCardStore,
  normalizeLoraCardStore,
  sameLoraCardStore,
  withGroupCard,
  withLoraCard,
  withoutGroupCard,
} from "./lora-cards.js";
import { prepareCardImage } from "./lora-card-image.js";

// One loader for the card store, shared by the generate page's LoRA modal and by
// the standalone asset page. They are separate documents, so each keeps its own
// copy and they tell each other about edits over the channel the mounted list
// already uses.
//
// Unlike the mounted library this is not on the generation path, so there is no
// lock to respect and no epoch guard: a card is edited by an explicit action in
// a dialog, never by a slider being dragged, and the last save of a rarely
// touched file is the right answer.

export const LORA_CARDS_MESSAGE = "workspace-lora-cards";
const SAVE_DELAY_MS = 250;

export function useLoraCards() {
  const [store, setStore] = useState(emptyLoraCardStore);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const storeRef = useRef(store);
  const channelRef = useRef(null);
  /** The debounce timer id, or null when nothing is waiting to be written. */
  const pendingRef = useRef(null);
  storeRef.current = store;

  useEffect(() => {
    let cancelled = false;
    fetch("/api/lora-cards", { cache: "no-store" })
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "无法读取 LoRA 卡片");
        return payload;
      })
      .then((payload) => {
        if (cancelled) return;
        setStore(normalizeLoraCardStore(payload.store).store);
        if (payload.reset) setError("LoRA 卡片文件无法识别，已重置为空；挂载库和组合未受影响。");
      })
      .catch((loadError) => !cancelled && setError(loadError.message))
      .finally(() => !cancelled && setReady(true));
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (typeof BroadcastChannel === "undefined") return undefined;
    const channel = new BroadcastChannel(LORA_SYNC_CHANNEL);
    channelRef.current = channel;
    channel.onmessage = (event) => {
      if (event.data?.type !== LORA_CARDS_MESSAGE) return;
      const incoming = normalizeLoraCardStore(event.data.store).store;
      setStore((current) => sameLoraCardStore(current, incoming) ? current : incoming);
    };
    return () => {
      channel.close();
      channelRef.current = null;
    };
  }, []);

  const save = useCallback(async ({ report = true } = {}) => {
    pendingRef.current = null;
    try {
      const response = await fetch("/api/lora-cards", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ store: storeRef.current }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法保存 LoRA 卡片");
      if (report) setError("");
    } catch (saveError) {
      if (report) setError(saveError.message);
    }
  }, []);

  const saveRef = useRef(save);
  saveRef.current = save;

  // Closing the standalone page moments after an edit must not lose it, so an
  // unmount flushes the debounce rather than cancelling it.
  useEffect(() => () => {
    if (pendingRef.current === null) return;
    window.clearTimeout(pendingRef.current);
    void saveRef.current({ report: false });
  }, []);

  const commit = useCallback((next) => {
    const normalized = normalizeLoraCardStore(next).store;
    if (sameLoraCardStore(storeRef.current, normalized)) return;
    storeRef.current = normalized;
    setStore(normalized);
    // Told first, saved second: the other window should reflect the edit within
    // a frame rather than after a round trip through the disk.
    channelRef.current?.postMessage({ type: LORA_CARDS_MESSAGE, store: normalized });
    if (pendingRef.current !== null) window.clearTimeout(pendingRef.current);
    pendingRef.current = window.setTimeout(() => void save(), SAVE_DELAY_MS);
  }, [save]);

  const patchLoraCard = useCallback((engine, value, patch) => {
    commit(withLoraCard(storeRef.current, engine, value, patch));
  }, [commit]);

  const patchGroupCard = useCallback((engine, groupId, patch) => {
    commit(withGroupCard(storeRef.current, engine, groupId, patch));
  }, [commit]);

  const removeGroupCard = useCallback((engine, groupId) => {
    commit(withoutGroupCard(storeRef.current, engine, groupId));
  }, [commit]);

  /**
   * Downscales and stores one picture, returning the id a card can point at.
   * The bytes are never referenced until the caller writes that id somewhere, so
   * an upload the user then cancels is collected by the next save.
   */
  const uploadCardImage = useCallback(async (source, options) => {
    const prepared = await prepareCardImage(source, options);
    const response = await fetch("/api/lora-cards/asset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataUrl: prepared.dataUrl }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "图片保存失败");
    return payload.id;
  }, []);

  return { store, ready, error, setError, patchLoraCard, patchGroupCard, removeGroupCard, uploadCardImage };
}
