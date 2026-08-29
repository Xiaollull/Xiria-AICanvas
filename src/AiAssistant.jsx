import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Check, CircleStop, History, LoaderCircle, MessageSquarePlus, Send, Settings2, Sparkles, Trash2, TriangleAlert, VenetianMask, WandSparkles } from "lucide-react";

import AssistantPersonaPanel from "./AssistantPersonaPanel.jsx";
import AssistantSettingsPanel from "./AssistantSettingsPanel.jsx";
import { providerProfile } from "./ai-assistant-providers.js";
import {
  ASSISTANT_ACTIVE_SESSION_KEY,
  ASSISTANT_CHANNEL_NAME,
  ASSISTANT_SESSIONS_CHANGED,
  normalizeConversation,
  optimizationRequest,
  segmentAssistantReply,
} from "./ai-assistant-protocol.js";
import {
  AssistantRequestError,
  createAssistantSession,
  deleteAssistantSession,
  fetchAssistantPersonas,
  fetchAssistantProfiles,
  fetchAssistantSession,
  fetchAssistantSessions,
  saveAssistantProfile,
  saveAssistantSession,
  streamAssistantChat,
} from "./assistant-client.js";

// Shared chat surface. The floating overlay and the standalone `/assistant` page both render this
// component and differ only in the chrome around it, so the two modes cannot drift apart.

// Superseded by server-side sessions. Read once so an existing conversation is carried into the
// session store instead of being discarded, then cleared.
const LEGACY_CONVERSATION_KEY = "xirai-assistant-conversation-v1";

const newId = (role) => `${role}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

function takeLegacyConversation() {
  try {
    const saved = window.localStorage.getItem(LEGACY_CONVERSATION_KEY);
    if (!saved) return [];
    window.localStorage.removeItem(LEGACY_CONVERSATION_KEY);
    return normalizeConversation(JSON.parse(saved)).messages;
  } catch {
    return [];
  }
}

function readActiveSessionId() {
  try {
    return window.localStorage.getItem(ASSISTANT_ACTIVE_SESSION_KEY) || "";
  } catch {
    return "";
  }
}

// Only the *pointer* stays in the browser; the conversations themselves live on disk. Sharing the
// key means popping out continues the same chat rather than opening a different one.
function writeActiveSessionId(id) {
  try {
    if (id) window.localStorage.setItem(ASSISTANT_ACTIVE_SESSION_KEY, id);
    else window.localStorage.removeItem(ASSISTANT_ACTIVE_SESSION_KEY);
  } catch {
    // A blocked storage only costs the remembered selection, not the history itself.
  }
}

function relativeTime(value) {
  const at = Date.parse(value);
  if (!Number.isFinite(at)) return "";
  const minutes = Math.floor((Date.now() - at) / 60000);
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.floor(hours / 24);
  return days < 30 ? `${days} 天前` : new Date(at).toLocaleDateString("zh-CN");
}

// Shown in place of an empty transcript. A blank chat surface gives no hint of what this assistant
// is for, and the standalone page is mostly blank space without them.
const CHAT_STARTERS = Object.freeze([
  "帮我写一段电影感人像的正向提示词",
  "把画面改成赛博朋克夜景，保留人物主体",
  "补充一组适合写实风格的负向提示词",
]);

// One list, two hosts: the overlay shows it as a view because it can be 360px wide, the standalone
// page shows it as a permanent rail. Sharing the markup keeps the two from drifting.
function SessionList({ sessions, activeId, busy, onOpen, onDelete }) {
  if (!sessions.length) {
    return (
      <div className="assistant-empty">
        <History size={24} />
        <p>还没有历史对话。</p>
        <p className="assistant-empty-hint">发送第一条消息后，本次对话会自动保存到这里。</p>
      </div>
    );
  }
  return sessions.map((session) => (
    <div className={`assistant-session-row ${session.id === activeId ? "active" : ""}`} key={session.id}>
      <button type="button" className="assistant-session-open" onClick={() => void onOpen(session.id)} disabled={busy}>
        <strong>{session.title}</strong>
        <small>{relativeTime(session.updatedAt)} · {session.messageCount} 条消息</small>
      </button>
      <button type="button" className="assistant-session-delete" onClick={() => void onDelete(session.id)} title="删除该对话" aria-label={`删除对话 ${session.title}`}>
        <Trash2 size={13} />
      </button>
    </div>
  ));
}

export default function AiAssistant({ onApplyPrompt, onStatusChange, promptSnapshot, variant = "overlay" }) {
  const isPage = variant === "page";
  const [view, setView] = useState("chat");
  const [messages, setMessages] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [sessionId, setSessionId] = useState("");
  const [sessionBusy, setSessionBusy] = useState(false);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  // The whole profile store, not one settings object: the settings page needs the list and the
  // chat surface needs whichever profile is active, and both come from the same reply.
  const [profileStore, setProfileStore] = useState(null);
  const [personas, setPersonas] = useState([]);
  const [personaDiagnostics, setPersonaDiagnostics] = useState([]);
  const [personaBusy, setPersonaBusy] = useState(false);
  const [formErrors, setFormErrors] = useState([]);
  const [notice, setNotice] = useState("");
  const [applied, setApplied] = useState("");

  const messagesRef = useRef(messages);
  const sessionIdRef = useRef("");
  const broadcastRef = useRef(null);
  const abortRef = useRef(null);
  const transcriptRef = useRef(null);
  const composerRef = useRef(null);

  messagesRef.current = messages;
  sessionIdRef.current = sessionId;

  const refreshSessions = useCallback(async (signal) => {
    try {
      setSessions(await fetchAssistantSessions(signal));
    } catch {
      // The picker is a convenience; failing to list must not take the current chat down.
    }
  }, []);

  // Persist settled turns only. A half-streamed bubble written to disk would come back after a
  // reload looking like a truncated answer the model actually produced.
  useEffect(() => {
    if (!sessionId || !messages.length || messages.some((message) => message.streaming)) return undefined;
    const timer = window.setTimeout(async () => {
      const saved = await saveAssistantSession(sessionId, { messages }).catch(() => null);
      if (!saved) return;
      setSessions((current) => current.map((entry) => entry.id === saved.id
        ? { ...entry, title: saved.title, updatedAt: saved.updatedAt, messageCount: saved.messages.length }
        : entry));
      broadcastRef.current?.postMessage({ type: ASSISTANT_SESSIONS_CHANGED, sessionId });
    }, 400);
    return () => window.clearTimeout(timer);
  }, [messages, sessionId]);

  // Keeps a second surface current. The other document owns its own streaming state, so a sync
  // that arrives mid-answer is ignored rather than overwriting a reply in flight.
  useEffect(() => {
    if (typeof window.BroadcastChannel !== "function") return undefined;
    const channel = new BroadcastChannel(ASSISTANT_CHANNEL_NAME);
    broadcastRef.current = channel;
    channel.onmessage = async (event) => {
      if (event.data?.type !== ASSISTANT_SESSIONS_CHANGED) return;
      await refreshSessions();
      const active = sessionIdRef.current;
      if (!active || event.data.sessionId !== active) return;
      if (messagesRef.current.some((message) => message.streaming)) return;
      const session = await fetchAssistantSession(active).catch(() => null);
      if (session) setMessages(session.messages);
    };
    return () => {
      channel.close();
      broadcastRef.current = null;
    };
  }, [refreshSessions]);

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        const [storePayload, personaPayload] = await Promise.all([
          fetchAssistantProfiles(controller.signal),
          fetchAssistantPersonas(controller.signal),
        ]);
        if (controller.signal.aborted) return;
        setProfileStore(storePayload);
        setPersonas(personaPayload.personas || []);
        setPersonaDiagnostics(personaPayload.diagnostics || []);
        // A client that has never been configured opens straight into settings: an empty chat box
        // wired to nothing is not a useful first screen.
        const live = storePayload.profiles?.find((profile) => profile.id === storePayload.activeId)?.settings;
        if (live && !live.hasApiKey && providerProfile(live.provider)?.requiresApiKey) setView("settings");
      } catch (error) {
        if (!controller.signal.aborted) setNotice(error.message || "无法读取 AI 助手配置");
      }

      try {
        let listed = await fetchAssistantSessions(controller.signal);
        // One-time carry-over of the pre-session transcript, so upgrading does not look like the
        // previous conversation was thrown away.
        const legacy = takeLegacyConversation();
        if (legacy.length) {
          const migrated = await createAssistantSession({ messages: legacy }, controller.signal);
          listed = [{ ...migrated, messageCount: migrated.messages.length }, ...listed];
        }
        if (controller.signal.aborted) return;
        setSessions(listed);
        const remembered = readActiveSessionId();
        const target = listed.find((entry) => entry.id === remembered) || listed[0];
        if (!target) return;
        const session = await fetchAssistantSession(target.id, controller.signal);
        if (controller.signal.aborted || !session) return;
        setSessionId(session.id);
        writeActiveSessionId(session.id);
        setMessages(session.messages);
      } catch (error) {
        if (!controller.signal.aborted) setNotice(error.message || "无法读取历史对话");
      }
    })();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const node = transcriptRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [messages]);

  // The live profile. Everything outside the settings page cares only about this one.
  const activeProfile = useMemo(
    () => profileStore?.profiles?.find((profile) => profile.id === profileStore.activeId) || null,
    [profileStore],
  );
  const settings = activeProfile?.settings || null;

  const configured = Boolean(settings) && (settings.hasApiKey || !providerProfile(settings.provider)?.requiresApiKey);

  // The character the next message will be sent as. A character the user deleted simply resolves
  // to nothing, exactly as the control plane's `selectPersona` does, so a stale id is never fatal.
  const activePersona = useMemo(
    () => personas.find((persona) => persona.id === settings?.personaId) || null,
    [personas, settings?.personaId],
  );

  // Every character mutation replies with the refreshed library, so the panel never has to refetch.
  const applyPersonaLibrary = useCallback((payload) => {
    setPersonas(payload.personas || []);
    setPersonaDiagnostics(payload.diagnostics || []);
  }, []);

  // The character library and the service configuration are separate things, so choosing a
  // character has to write it onto the live profile — that is where `personaId` is stored, and
  // where the chat route reads it.
  const useCharacterForChat = useCallback(async (personaId) => {
    if (!activeProfile || personaBusy) return;
    setPersonaBusy(true);
    try {
      const { provider, baseUrl, model, strength } = activeProfile.settings;
      setProfileStore(await saveAssistantProfile(activeProfile.id, {
        settings: { provider, baseUrl, model, strength, personaId },
      }));
    } catch (error) {
      setNotice(error.message || "无法切换角色");
    } finally {
      setPersonaBusy(false);
    }
  }, [activeProfile, personaBusy]);

  // The chrome around this component shows which service is live. Reporting it upward keeps the
  // settings owned here rather than having the window and the page each fetch them again. `ready`
  // exists so the chrome stays blank until the answer is known: settings load asynchronously, and
  // "not configured" is an alarming thing to flash at someone who configured it long ago.
  useEffect(() => {
    onStatusChange?.({ ready: Boolean(settings), model: settings?.model || "", provider: settings?.provider || "", profile: activeProfile?.name || "", configured });
  }, [activeProfile, configured, onStatusChange, settings]);

  const stop = useCallback(() => { abortRef.current?.abort(); }, []);

  // Sessions are created on the first message rather than on open, so browsing the assistant and
  // closing it again does not litter the history with empty conversations.
  const ensureSession = useCallback(async () => {
    if (sessionIdRef.current) return sessionIdRef.current;
    const created = await createAssistantSession({});
    sessionIdRef.current = created.id;
    setSessionId(created.id);
    writeActiveSessionId(created.id);
    setSessions((current) => [{ ...created, messageCount: 0 }, ...current]);
    return created.id;
  }, []);

  const send = useCallback(async (text) => {
    const trimmed = text.trim();
    if (!trimmed || streaming) return;
    try {
      await ensureSession();
    } catch {
      setNotice("无法新建对话，历史记录可能不可写。");
      return;
    }
    const userMessage = { id: newId("user"), role: "user", content: trimmed, reasoning: "", failed: false };
    const replyId = newId("assistant");
    const history = [...messagesRef.current, userMessage];
    setMessages([...history, { id: replyId, role: "assistant", content: "", reasoning: "", failed: false, streaming: true }]);
    setInput("");
    setStreaming(true);
    setNotice("");
    const controller = new AbortController();
    abortRef.current = controller;
    const patch = (changes) => setMessages((current) => current.map((message) => message.id === replyId ? { ...message, ...changes } : message));
    try {
      const { content, reasoning } = await streamAssistantChat({
        history,
        signal: controller.signal,
        onDelta: (progress) => patch({ content: progress.content, reasoning: progress.reasoning }),
      });
      patch({ content, reasoning, streaming: false });
    } catch (error) {
      const aborted = controller.signal.aborted || error?.name === "AbortError";
      patch({ content: aborted ? "已停止生成。" : (error?.message || "请求失败"), streaming: false, failed: true });
      // A configuration fault is only fixable in the settings view, so surface it there directly.
      if (!aborted && error instanceof AssistantRequestError && error.errors.length) {
        setFormErrors(error.errors);
        setView("settings");
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }, [ensureSession, streaming]);

  // The original one-shot behaviour of the old optimise button, now expressed as a conversation
  // turn so the user can keep refining the result instead of only accepting or discarding it.
  const optimizeCurrentPrompt = useCallback(() => {
    const snapshot = promptSnapshot?.() || { positive: "", negative: "" };
    if (!snapshot.positive.trim()) {
      setNotice("正向提示词为空，请先输入内容或直接在下方描述你想要的画面。");
      return;
    }
    void send(optimizationRequest(snapshot.positive, snapshot.negative));
  }, [promptSnapshot, send]);

  const apply = useCallback((segment, messageId) => {
    onApplyPrompt?.(segment.type === "negative" ? { negative: segment.content } : { positive: segment.content });
    setApplied(`${messageId}:${segment.type}:${segment.content.length}`);
    window.setTimeout(() => setApplied(""), 1600);
  }, [onApplyPrompt]);

  // "New chat" leaves the current conversation on disk and starts an empty one. The empty session
  // itself is not created until the first message, so repeatedly pressing it costs nothing.
  const startNewSession = useCallback(() => {
    stop();
    setMessages([]);
    setSessionId("");
    sessionIdRef.current = "";
    writeActiveSessionId("");
    setNotice("");
    setView("chat");
  }, [stop]);

  const openSession = useCallback(async (id) => {
    if (sessionBusy) return;
    setSessionBusy(true);
    stop();
    try {
      const session = await fetchAssistantSession(id);
      if (!session) {
        // Another tab deleted it while the picker was open.
        setSessions((current) => current.filter((entry) => entry.id !== id));
        setNotice("该对话已被删除。");
        return;
      }
      setSessionId(session.id);
      sessionIdRef.current = session.id;
      writeActiveSessionId(session.id);
      setMessages(session.messages);
      setNotice("");
      setView("chat");
    } catch (error) {
      setNotice(error.message || "无法读取该对话");
    } finally {
      setSessionBusy(false);
    }
  }, [sessionBusy, stop]);

  const removeSession = useCallback(async (id) => {
    try {
      await deleteAssistantSession(id);
    } catch (error) {
      setNotice(error.message || "无法删除该对话");
      return;
    }
    setSessions((current) => current.filter((entry) => entry.id !== id));
    if (sessionIdRef.current === id) {
      setMessages([]);
      setSessionId("");
      sessionIdRef.current = "";
      writeActiveSessionId("");
    }
    broadcastRef.current?.postMessage({ type: ASSISTANT_SESSIONS_CHANGED, sessionId: id });
  }, []);

  const composerKeyDown = (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.ctrlKey || event.metaKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    void send(input);
  };

  const sessionList = (
    <SessionList sessions={sessions} activeId={sessionId} busy={sessionBusy} onOpen={openSession} onDelete={removeSession} />
  );

  return (
    <div className={`assistant-body assistant-${variant}`}>
      {/* The standalone page has the width for a permanent rail; the floating window does not, and
          reaches the same list through a tab instead. */}
      {isPage && (
        <aside className="assistant-rail">
          <button type="button" className="assistant-rail-new" onClick={startNewSession}><MessageSquarePlus size={13} />新建对话</button>
          <div className="assistant-rail-label">历史对话{sessions.length > 0 && <b>{sessions.length}</b>}</div>
          <div className="assistant-sessions">{sessionList}</div>
        </aside>
      )}

      <div className="assistant-main">
        <div className="assistant-toolbar">
          <div className="assistant-tabs">
            <button type="button" className={view === "chat" ? "active" : ""} onClick={() => setView("chat")}><Sparkles size={13} />对话</button>
            {!isPage && <button type="button" className={view === "sessions" ? "active" : ""} onClick={() => { void refreshSessions(); setView("sessions"); }}><History size={13} />历史{sessions.length > 0 && <b>{sessions.length}</b>}</button>}
            <button type="button" className={view === "personas" ? "active" : ""} onClick={() => setView("personas")}><VenetianMask size={13} />角色</button>
            <button type="button" className={view === "settings" ? "active" : ""} onClick={() => setView("settings")}><Settings2 size={13} />服务配置</button>
          </div>
          <span className="assistant-spacer" />
          {!isPage && <button type="button" className="assistant-ghost" onClick={startNewSession} disabled={!sessionId && !messages.length} title="新建对话" aria-label="新建对话"><MessageSquarePlus size={13} /></button>}
        </div>

        {view === "sessions" ? (
          <div className="assistant-sessions">
            {sessionList}
            {notice && <p className="assistant-notice">{notice}</p>}
          </div>
        ) : view === "personas" ? (
          <AssistantPersonaPanel
            personas={personas}
            diagnostics={personaDiagnostics}
            activeId={settings?.personaId || ""}
            busy={personaBusy}
            onLibraryChange={applyPersonaLibrary}
            onUsePersona={useCharacterForChat}
          />
        ) : view === "settings" ? (
          <AssistantSettingsPanel
            store={profileStore}
            onStoreChange={setProfileStore}
            personas={personas}
            personaDiagnostics={personaDiagnostics}
            errors={formErrors}
            onErrorsChange={setFormErrors}
          />
        ) : (
          <>
            <div className="assistant-transcript" ref={transcriptRef}>
              {!messages.length && (
                <div className="assistant-empty">
                  <WandSparkles size={isPage ? 30 : 26} />
                  <strong>开始一段对话</strong>
                  <p>{activePersona?.description || "描述你想要的画面，或让助手改写当前的正向提示词。给出的提示词会显示为可一键应用的卡片。"}</p>
                  {/* Which character is answering, and its own suggested openers when it has them.
                      A character with no starters falls back to the generic three. */}
                  {activePersona && (
                    <button type="button" className="assistant-empty-persona" onClick={() => setView("personas")} title="打开角色设置">
                      {activePersona.name}
                    </button>
                  )}
                  <div className="assistant-starters">
                    {(activePersona?.starters?.length ? activePersona.starters : CHAT_STARTERS).map((starter) => (
                      <button type="button" key={starter} disabled={!configured} onClick={() => { setInput(starter); composerRef.current?.focus(); }}>{starter}</button>
                    ))}
                  </div>
                </div>
              )}
              {messages.map((message) => message.role === "user" ? (
                <div className="assistant-message assistant-user" key={message.id}><p>{message.content}</p></div>
              ) : (
                <div className={`assistant-message assistant-reply ${message.failed ? "failed" : ""}`} key={message.id}>
                  {message.reasoning && <details className="assistant-reasoning"><summary>推理过程</summary><pre>{message.reasoning}</pre></details>}
                  {message.streaming && !message.content && <p className="assistant-thinking"><LoaderCircle size={13} className="spin" />正在思考...</p>}
                  {segmentAssistantReply(message.content).map((segment, index) => segment.type === "text" ? (
                    <p key={index}>{segment.content}</p>
                  ) : (
                    <div className={`assistant-prompt-card ${segment.type}`} key={index}>
                      <header><span>{segment.type === "negative" ? "负向提示词" : "正向提示词"}</span></header>
                      <code>{segment.content}</code>
                      <button type="button" onClick={() => apply(segment, message.id)}>
                        {applied === `${message.id}:${segment.type}:${segment.content.length}` ? <><Check size={12} />已应用</> : <>应用到{segment.type === "negative" ? "负向" : "正向"}提示词</>}
                      </button>
                    </div>
                  ))}
                  {message.failed && <p className="assistant-failure"><TriangleAlert size={13} />{message.content}</p>}
                </div>
              ))}
            </div>

            {/* Textarea and its controls share one bordered box: two stacked rows cost a third of the
                vertical space in a 380px-tall window, and the send button was detached from the field
                it belongs to. */}
            <div className="assistant-composer">
              <div className="assistant-composer-box">
                <textarea
                  ref={composerRef}
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={composerKeyDown}
                  placeholder={configured ? "描述画面，或提出修改意见..." : "请先在「服务配置」中填写服务地址与 API Key"}
                  spellCheck={false}
                  rows={2}
                  disabled={!configured}
                />
                <div className="assistant-composer-actions">
                  <button type="button" onClick={optimizeCurrentPrompt} disabled={streaming || !configured}><WandSparkles size={13} />优化当前提示词</button>
                  {!configured && <span className="assistant-unconfigured"><TriangleAlert size={12} />尚未配置服务</span>}
                  <span className="assistant-spacer" />
                  <span className="assistant-composer-hint">Enter 发送 · Shift+Enter 换行</span>
                  {streaming
                    ? <button type="button" className="assistant-stop" onClick={stop} title="停止生成" aria-label="停止生成"><CircleStop size={15} /></button>
                    : <button type="button" className="assistant-send" onClick={() => void send(input)} disabled={!input.trim() || !configured} title="发送" aria-label="发送"><Send size={15} /></button>}
                </div>
              </div>
              {notice && <p className="assistant-notice">{notice}</p>}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
