import { useCallback, useMemo, useState } from "react";
import { BookOpen, Check, Copy, Eye, EyeOff, Lock, Plus, Save, Trash2, TriangleAlert, UserRound, UserRoundPlus } from "lucide-react";

import { composeSystemPrompt } from "./ai-assistant-protocol.js";
import {
  MAXIMUM_PERSONA_RULE,
  MAXIMUM_PERSONA_RULES,
  MAXIMUM_PERSONA_SPECIALTIES,
  MAXIMUM_PERSONA_STARTERS,
  MAXIMUM_PERSONA_NAME,
  MAXIMUM_PERSONA_DESCRIPTION,
  PERSONA_ATTRIBUTES,
  draftAssistantPersona,
  hasPersonaAttributes,
  normalizeAssistantPersona,
} from "./assistant-persona.js";
import {
  AssistantRequestError,
  createAssistantPersona,
  deleteAssistantPersona,
  saveAssistantPersona,
} from "./assistant-client.js";

// The character configuration interface: a library of AI characters on the left, an editor for the
// selected one on the right.
//
// A character is not a second prompt box. Its attributes, specialties and behaviour rules are
// rendered into the *same* system message its prose goes into, by `composeSystemPrompt` — the very
// function the control plane calls at request time. That is why this panel previews the composed
// result rather than describing it: what the preview shows is byte-for-byte what the provider
// receives, including the output protocol the control plane always appends.
//
// Shipped characters are read-only. An app update replaces `assistant/personas/` wholesale, so a
// built-in that could be edited would silently lose the edit; 复制 is offered instead.

// Three list-shaped fields — specialties, behaviour rules, suggested openers — share one editor.
// A textarea rather than per-row inputs with buttons: it triples less markup, and people author
// these by pasting a list they already have rather than by clicking 新增 twelve times.
function LineListField({ label, hint, value, maximum, itemLimit, placeholder, rows, onChange }) {
  const lines = value.split("\n").map((line) => line.trim()).filter(Boolean);
  const over = lines.length > maximum;
  return (
    <label className="assistant-field">
      <span>{label}<b className={over ? "over" : ""}>{lines.length}/{maximum}</b></span>
      <textarea
        className="assistant-persona-lines"
        rows={rows}
        spellCheck={false}
        placeholder={placeholder}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
      <small>{hint}每行一条，最多 {maximum} 条，单条不超过 {itemLimit} 字{over ? "；超出的部分保存时会被丢弃。" : "。"}</small>
    </label>
  );
}

function PersonaRow({ persona, active, dirty, busy, confirming, onOpen, onCopy, onRemove }) {
  return (
    <div className={`assistant-lib-row ${active ? "active" : ""}`}>
      <button type="button" className="assistant-lib-open" onClick={() => onOpen(persona.id)} disabled={busy}>
        <strong>
          <UserRound size={12} className="assistant-persona-mark" />
          {persona.name}
          {dirty && <em className="assistant-lib-dirty" title="有尚未保存的修改">未保存</em>}
        </strong>
        <small>
          {persona.builtIn && <i className="assistant-persona-badge"><Lock size={9} />内置</i>}
          {persona.description || (hasPersonaAttributes(persona) ? "已设定属性与行为" : "仅角色提示词")}
        </small>
      </button>
      <div className="assistant-lib-tools">
        <button type="button" onClick={() => onCopy(persona.id)} disabled={busy} title="复制为一个可编辑的角色" aria-label={`复制角色 ${persona.name}`}>
          <Copy size={12} />
        </button>
        <button
          type="button"
          className={confirming ? "confirming" : ""}
          onClick={() => onRemove(persona.id)}
          disabled={busy || persona.builtIn}
          title={persona.builtIn ? "内置角色不可删除，可复制后编辑" : confirming ? "再次点击确认删除" : "删除这个角色"}
          aria-label={`删除角色 ${persona.name}`}
        >
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  );
}

const editorFromPersona = (persona) => ({
  name: persona?.name || "",
  description: persona?.description || "",
  traits: { ...(persona?.traits || {}) },
  specialties: (persona?.specialties || []).join("\n"),
  rules: (persona?.rules || []).join("\n"),
  systemPrompt: persona?.systemPrompt || "",
  starters: (persona?.starters || []).join("\n"),
});

// The editor holds the three list fields as text, so the record has to be rebuilt before it can be
// previewed or saved. One function does both, which is what keeps the preview honest.
const personaFromEditor = (draft, { id, builtIn } = {}) => normalizeAssistantPersona({
  ...draft,
  specialties: draft.specialties.split("\n"),
  rules: draft.rules.split("\n"),
  starters: draft.starters.split("\n"),
}, { id, builtIn });

const sameEditor = (left, right) => ["name", "description", "specialties", "rules", "systemPrompt", "starters"]
  .every((field) => left[field] === right[field])
  && PERSONA_ATTRIBUTES.every((entry) => (left.traits?.[entry.id] || "") === (right.traits?.[entry.id] || ""));

export default function AssistantPersonaPanel({ personas, diagnostics, activeId, busy: hostBusy, onLibraryChange, onUsePersona }) {
  const [selectedId, setSelectedId] = useState("");
  const [editors, setEditors] = useState({});
  const [confirmingDelete, setConfirmingDelete] = useState("");
  const [showPreview, setShowPreview] = useState(false);
  const [errors, setErrors] = useState([]);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);

  // Falls back to the character the assistant is actually using, so opening this tab lands on
  // something relevant rather than on whatever happens to sort first.
  const selected = useMemo(
    () => personas.find((persona) => persona.id === selectedId)
      || personas.find((persona) => persona.id === activeId)
      || personas[0]
      || null,
    [activeId, personas, selectedId],
  );
  const stored = useMemo(() => selected ? editorFromPersona(selected) : null, [selected]);
  const draft = (selected && editors[selected.id]) || stored;

  const dirtyIds = useMemo(() => new Set(personas
    .filter((persona) => editors[persona.id] && !sameEditor(editors[persona.id], editorFromPersona(persona)))
    .map((persona) => persona.id)), [editors, personas]);
  const dirty = Boolean(selected && dirtyIds.has(selected.id));

  // The exact system message this character would produce, protocol included. Recomputed from the
  // draft rather than from the stored record, so the preview follows the form as it is typed.
  const preview = useMemo(
    () => draft && selected ? composeSystemPrompt(personaFromEditor(draft, { id: selected.id, builtIn: selected.builtIn })) : "",
    [draft, selected],
  );

  const patchDraft = useCallback((changes) => {
    if (!selected) return;
    setEditors((current) => ({ ...current, [selected.id]: { ...(current[selected.id] || editorFromPersona(selected)), ...changes } }));
  }, [selected]);

  const patchTrait = useCallback((id, value) => {
    if (!selected) return;
    setEditors((current) => {
      const base = current[selected.id] || editorFromPersona(selected);
      return { ...current, [selected.id]: { ...base, traits: { ...base.traits, [id]: value } } };
    });
  }, [selected]);

  const run = useCallback(async (work, { failure } = {}) => {
    if (busy || hostBusy) return null;
    setBusy(true);
    setErrors([]);
    setNotice("");
    try {
      return await work();
    } catch (error) {
      setErrors(error instanceof AssistantRequestError ? error.errors : []);
      setNotice(error.message || failure || "操作失败");
      return null;
    } finally {
      setBusy(false);
    }
  }, [busy, hostBusy]);

  const applyLibrary = useCallback((payload, { clearEditorFor, select } = {}) => {
    onLibraryChange(payload);
    if (select) setSelectedId(select);
    if (clearEditorFor) setEditors((current) => {
      const rest = { ...current };
      delete rest[clearEditorFor];
      return rest;
    });
  }, [onLibraryChange]);

  const openPersona = useCallback((id) => {
    setConfirmingDelete("");
    setErrors([]);
    setNotice("");
    setSelectedId(id);
  }, []);

  const addPersona = useCallback(() => {
    void run(async () => {
      const payload = await createAssistantPersona(draftAssistantPersona());
      applyLibrary(payload, { select: payload.createdId });
      setNotice("已新建角色，填写角色提示词或属性后保存。");
    }, { failure: "无法新建角色" });
  }, [applyLibrary, run]);

  // The only way to start from a built-in. The server copies it, so the shipped file is never the
  // thing being edited.
  const copyPersona = useCallback((id) => {
    void run(async () => {
      const payload = await createAssistantPersona({ fromId: id });
      applyLibrary(payload, { select: payload.createdId });
      setNotice("已复制为可编辑的角色。");
    }, { failure: "无法复制该角色" });
  }, [applyLibrary, run]);

  const removePersona = useCallback((id) => {
    if (confirmingDelete !== id) {
      setConfirmingDelete(id);
      setNotice("再次点击删除按钮以确认删除该角色。");
      window.setTimeout(() => setConfirmingDelete((current) => current === id ? "" : current), 4000);
      return;
    }
    setConfirmingDelete("");
    void run(async () => {
      const payload = await deleteAssistantPersona(id);
      applyLibrary(payload, { clearEditorFor: id, select: "" });
      setNotice("角色已删除。");
    }, { failure: "无法删除该角色" });
  }, [applyLibrary, confirmingDelete, run]);

  const persist = useCallback(() => {
    if (!selected || !draft || selected.builtIn) return;
    void run(async () => {
      const payload = await saveAssistantPersona(selected.id, personaFromEditor(draft, { id: selected.id }));
      applyLibrary(payload, { clearEditorFor: selected.id, select: selected.id });
      setNotice("角色已保存。");
    }, { failure: "无法保存角色" });
  }, [applyLibrary, draft, run, selected]);

  if (!personas.length || !selected || !draft) {
    return (
      <div className="assistant-lib">
        <div className="assistant-lib-form">
          <p className="assistant-hint">正在读取角色…</p>
        </div>
      </div>
    );
  }

  const readOnly = selected.builtIn;
  const inUse = selected.id === activeId;

  return (
    <div className="assistant-lib">
      <aside className="assistant-lib-rail">
        <div className="assistant-lib-head">
          <span>角色库<b>{personas.length}</b></span>
          <button type="button" className="assistant-lib-add" onClick={addPersona} disabled={busy} title="新建一个角色">
            <Plus size={12} />新建
          </button>
        </div>
        <div className="assistant-lib-list">
          {personas.map((persona) => (
            <PersonaRow
              key={persona.id}
              persona={persona}
              active={persona.id === selected.id}
              dirty={dirtyIds.has(persona.id)}
              busy={busy}
              confirming={confirmingDelete === persona.id}
              onOpen={openPersona}
              onCopy={copyPersona}
              onRemove={removePersona}
            />
          ))}
        </div>
        {diagnostics.length > 0 && (
          <p className="assistant-hint assistant-hint-warn"><TriangleAlert size={12} />有 {diagnostics.length} 个角色文件无法解析，已跳过。</p>
        )}
        <p className="assistant-lib-foot">角色决定助手的说话方式与行为；对话使用的是「服务配置」中选定的角色。</p>
      </aside>

      <div className="assistant-lib-form assistant-persona-form">
        <div className="assistant-persona-head">
          <span className="assistant-persona-title">
            <UserRound size={16} className="assistant-persona-mark" />
            <strong>{draft.name || "未命名角色"}</strong>
            {readOnly && <em className="assistant-persona-badge"><Lock size={9} />内置角色，不可修改</em>}
            {inUse && <em className="assistant-lib-live"><Check size={11} />对话中</em>}
          </span>
          <span className="assistant-spacer" />
          {!inUse && (
            <button type="button" className="assistant-persona-use" onClick={() => onUsePersona(selected.id)} disabled={busy || hostBusy}>
              <UserRoundPlus size={12} />用于对话
            </button>
          )}
        </div>

        {readOnly && (
          <p className="assistant-hint assistant-hint-block">
            内置角色随应用一起更新，直接修改会在下次更新时丢失。点击列表中的复制按钮，得到一份可以自由编辑的副本。
          </p>
        )}

        {/* Two panes on the standalone page: what the character *is* on the left, what it will
            actually send on the right. The editor otherwise sat in a measured 860px column with
            empty gutters either side of a screen wide enough to show both at once. The window
            collapses this to one column — it is 360px at its narrowest. */}
        <div className="assistant-persona-columns">
          <div className="assistant-persona-column">
            <label className="assistant-field">
              <span>角色名称</span>
              <input
                spellCheck={false}
                maxLength={MAXIMUM_PERSONA_NAME}
                placeholder="例如：赛博朋克摄影师"
                value={draft.name}
                onChange={(event) => patchDraft({ name: event.target.value })}
                disabled={readOnly}
              />
              <small>显示在角色库与角色选择器中。</small>
            </label>

            <label className="assistant-field">
              <span>一句话简介</span>
              <input
                spellCheck={false}
                maxLength={MAXIMUM_PERSONA_DESCRIPTION}
                placeholder="偏爱霓虹夜景与胶片颗粒的摄影指导"
                value={draft.description}
                onChange={(event) => patchDraft({ description: event.target.value })}
                disabled={readOnly}
              />
              {/* Stated here because it is genuinely surprising: everything else on this form travels
                  upstream, and this one field deliberately does not. */}
              <small>只用于你自己辨认角色，不会发送给模型。</small>
            </label>

            <div className="assistant-persona-traits">
              {PERSONA_ATTRIBUTES.map((entry) => (
                <label className="assistant-field" key={entry.id}>
                  <span>{entry.label}</span>
                  <select value={draft.traits?.[entry.id] || ""} onChange={(event) => patchTrait(entry.id, event.target.value)} disabled={readOnly}>
                    {entry.choices.map((choice) => <option key={choice.value || "unset"} value={choice.value}>{choice.label}</option>)}
                  </select>
                  <small>{entry.hint}</small>
                </label>
              ))}
            </div>

            <LineListField
              label="擅长领域"
              hint="会作为一行写进系统提示词。"
              value={draft.specialties}
              maximum={MAXIMUM_PERSONA_SPECIALTIES}
              itemLimit={40}
              rows={3}
              placeholder={"电影感人像\n赛博朋克夜景"}
              onChange={(value) => patchDraft({ specialties: value })}
            />

            <LineListField
              label="行为准则"
              hint="按顺序编号后写进系统提示词，用来约束它怎么做事。"
              value={draft.rules}
              maximum={MAXIMUM_PERSONA_RULES}
              itemLimit={MAXIMUM_PERSONA_RULE}
              rows={4}
              placeholder={"先给提示词，再用一句话说明改动重点\n不要在提示词里堆砌无意义的画质词"}
              onChange={(value) => patchDraft({ rules: value })}
            />
          </div>

          <div className="assistant-persona-column">
            <label className="assistant-field assistant-persona-prompt">
              <span>角色提示词<b>{draft.systemPrompt.length}</b></span>
              <textarea
                rows={8}
                spellCheck={false}
                placeholder="你是一位……&#10;&#10;可以在这里写更完整的设定：身份、工作方式、偏好、禁忌。"
                value={draft.systemPrompt}
                onChange={(event) => patchDraft({ systemPrompt: event.target.value })}
                disabled={readOnly}
              />
              <small>自由发挥的部分。不需要再写 ```prompt 输出协议，控制面会自动追加；重复声明反而会让「应用到提示词」失效。</small>
            </label>

            <LineListField
              label="推荐开场白"
              hint="选中该角色时，显示在空对话里作为快捷入口。"
              value={draft.starters}
              maximum={MAXIMUM_PERSONA_STARTERS}
              itemLimit={80}
              rows={3}
              placeholder={"帮我写一段霓虹夜景的正向提示词\n把画面改成雨后街道，保留人物主体"}
              onChange={(value) => patchDraft({ starters: value })}
            />

            <div className="assistant-persona-preview">
              <button type="button" className="assistant-persona-preview-toggle" onClick={() => setShowPreview((current) => !current)}>
                {showPreview ? <EyeOff size={12} /> : <Eye size={12} />}
                {showPreview ? "收起系统提示词预览" : "预览实际发送的系统提示词"}
                <b>{preview.length} 字</b>
              </button>
              {showPreview && (
                <>
                  <pre>{preview}</pre>
                  <small><BookOpen size={11} />这就是每次对话开头发给模型的完整内容，末尾的输出协议由控制面自动追加，无法关闭。</small>
                </>
              )}
            </div>
          </div>
        </div>

        {errors.length > 0 && (
          <ul className="assistant-errors">{errors.map((entry) => <li key={`${entry.field}-${entry.code}`}>{entry.message}</li>)}</ul>
        )}
        {!readOnly && (
          <div className="assistant-settings-actions">
            <button type="button" className="primary" onClick={persist} disabled={busy || !dirty}>
              <Save size={13} />{dirty ? "保存角色" : "已保存"}
            </button>
          </div>
        )}
        {notice && <p className="assistant-notice">{notice}</p>}
      </div>
    </div>
  );
}
