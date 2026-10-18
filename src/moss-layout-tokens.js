// The layout's token count, from the service's own tokenizer (QPlan2/MOSS/P2 §5.4).
//
// The browser cannot count Qwen tokens, and an estimate that disagrees with the server's 4096 refusal
// is worse than none, so the card and the Studio show what `/api/inference/moss/layout/compile`
// reports for the base canvas and every Hires canvas. The request is debounced and only re-sent when
// the compiled prompt or the stage list changes.
import { useEffect, useRef, useState } from "react";
import { MOSS_LAYOUT_MAX_TOKENS } from "./moss-layout.js";

const DEBOUNCE_MS = 450;

const IDLE = Object.freeze({ status: "idle", tokens: null, stages: [], maxTokens: MOSS_LAYOUT_MAX_TOKENS, error: "" });

/** The largest count across the base and the stages, or null while unknown. */
export function mossLayoutPeakTokens(state) {
  if (!state || state.status !== "ready" || !Number.isFinite(state.tokens)) return null;
  return Math.max(state.tokens, ...state.stages.map((stage) => stage.tokens).filter(Number.isFinite));
}

/**
 * `{status, tokens, stages, maxTokens, error}` for the given compiled prompt. `status` is
 * idle (nothing to count), counting, ready, or error (the service could not count; the request is
 * still checked when it is submitted).
 */
export function useMossLayoutTokens({ active, payload, prompt, stages }) {
  const [state, setState] = useState(IDLE);
  const cache = useRef(new Map());
  const stageKey = JSON.stringify((stages || []).map((stage) => [stage.width, stage.height]));
  const key = active && prompt ? `${stageKey}\u0000${prompt}` : "";
  const payloadRef = useRef(payload);
  payloadRef.current = payload;

  useEffect(() => {
    if (!key) { setState(IDLE); return undefined; }
    const known = cache.current.get(key);
    if (known) { setState(known); return undefined; }
    setState((current) => ({ ...current, status: "counting", error: "" }));
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const response = await fetch("/api/inference/moss/layout/compile", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({ layout: payloadRef.current, stages: JSON.parse(stageKey) }),
        });
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`);
        const next = body.valid
          ? { status: "ready", tokens: body.tokens, stages: Array.isArray(body.stages) ? body.stages : [], maxTokens: body.max_tokens || MOSS_LAYOUT_MAX_TOKENS, error: "" }
          : { ...IDLE, status: "error", error: body.issues?.[0] ? `${body.issues[0].path} ${body.issues[0].message}` : "服务认为布局无效" };
        if (next.status === "ready") {
          cache.current.set(key, next);
          if (cache.current.size > 64) cache.current.delete(cache.current.keys().next().value);
        }
        setState(next);
      } catch (error) {
        if (controller.signal.aborted) return;
        setState({ ...IDLE, status: "error", error: error?.message || "无法连接推理服务" });
      }
    }, DEBOUNCE_MS);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [key, stageKey]);

  return state;
}
