import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft, Bot, Check } from "lucide-react";

import AiAssistant from "./AiAssistant.jsx";
import { ASSISTANT_APPLY_PROMPT, ASSISTANT_CHANNEL_NAME, readPromptSnapshot } from "./ai-assistant-protocol.js";
import { DEFAULT_THEME, applyThemeToDocument, loadThemeState } from "./theme.js";

// Standalone `/assistant` route: the pop-out target of the floating window.
//
// This document has no workspace of its own, so an applied prompt has to reach the tab that does.
// BroadcastChannel carries it there; if no workspace tab is listening the user is told so rather
// than being left thinking the prompt landed somewhere.

export default function AiAssistantPage() {
  const [applied, setApplied] = useState("");
  const [status, setStatus] = useState({ ready: false, model: "", configured: false });
  const channelRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/ui-state", { cache: "no-store" })
      .then(async (response) => response.ok ? response.json() : null)
      .then((payload) => { if (!cancelled) applyThemeToDocument(loadThemeState(payload?.state?.theme)); })
      .catch(() => { if (!cancelled) applyThemeToDocument({ ...DEFAULT_THEME }); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (typeof window.BroadcastChannel !== "function") return undefined;
    channelRef.current = new BroadcastChannel(ASSISTANT_CHANNEL_NAME);
    return () => {
      channelRef.current?.close();
      channelRef.current = null;
    };
  }, []);

  const applyPrompt = useCallback((prompt) => {
    if (!channelRef.current) {
      setApplied("当前浏览器不支持跨标签页同步，请手动复制提示词。");
      return;
    }
    channelRef.current.postMessage({ type: ASSISTANT_APPLY_PROMPT, ...prompt });
    setApplied(prompt.negative ? "负向提示词已发送到工作区标签页。" : "正向提示词已发送到工作区标签页。");
    window.setTimeout(() => setApplied(""), 2600);
  }, []);

  // A single app frame rather than a centred card: this route owns the whole viewport, and a fixed
  // content column left most of a wide screen empty while the chat itself stayed cramped.
  return (
    <main className="assistant-page-shell">
      <header className="assistant-page-top">
        <a className="assistant-page-back" href="/"><ArrowLeft size={13} />返回工作区</a>
        <span className="assistant-page-mark"><Bot size={16} /><h1>AI 助手</h1></span>
        {/* The saved configuration in use, not just the model: with several profiles stored, the
            model id alone does not say which of them the next message will go to. */}
        {status.ready && <span className={`assistant-page-model ${status.configured ? "" : "off"}`} title={status.model}>{status.profile || status.model || "尚未配置服务"}</span>}
        <span className="assistant-spacer" />
        {applied && <span className="assistant-page-applied"><Check size={12} />{applied}</span>}
      </header>
      <section className="assistant-page-main">
        {/* This page has no prompt boxes of its own, so "optimise current prompt" reads the mirror
            the workspace tab keeps up to date rather than an empty local state. */}
        <AiAssistant variant="page" onApplyPrompt={applyPrompt} onStatusChange={setStatus} promptSnapshot={() => readPromptSnapshot(window.localStorage)} />
      </section>
    </main>
  );
}
