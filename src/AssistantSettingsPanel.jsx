import { useCallback, useMemo, useState } from "react";
import { Check, Copy, Eye, EyeOff, Key, List, LoaderCircle, Pencil, Plus, RefreshCw, Trash2, TriangleAlert } from "lucide-react";

import { AI_PROVIDERS, providerProfile, resolveStrengthParameter } from "./ai-assistant-providers.js";
import { MAXIMUM_PROFILES, MAXIMUM_PROFILE_NAME } from "./assistant-profiles.js";
import {
  AssistantRequestError,
  activateAssistantProfile,
  createAssistantProfile,
  deleteAssistantProfile,
  duplicateAssistantProfile,
  fetchAssistantModels,
  saveAssistantProfile,
  testAssistantConnection,
} from "./assistant-client.js";

// The assistant's settings page: a list of named service configurations beside an editor for the
// selected one.
//
// Selecting a profile *is* activating it. The chat surface, the model-list fetch and the connection
// probe all read "the live configuration", so a page where you could edit a profile that was not
// the one being tested would need every one of those routes to grow a profile parameter, and would
// still leave the user guessing which service the chat was about to use.
//
// Both hosts render this component. The floating window is 360 px at its narrowest, so the list
// sits above the form there and beside it on `/assistant`; that is a CSS decision, not a fork.

const editorFromProfile = (profile) => ({
  name: profile?.name || "",
  provider: profile?.settings?.provider || "",
  baseUrl: profile?.settings?.baseUrl || "",
  model: profile?.settings?.model || "",
  strength: profile?.settings?.strength ?? null,
  personaId: profile?.settings?.personaId || "",
  apiKey: "",
});

const sameEditor = (left, right) => ["name", "provider", "baseUrl", "model", "personaId", "apiKey"]
  .every((field) => left[field] === right[field]) && left.strength === right.strength;

function ErrorList({ errors }) {
  if (!errors?.length) return null;
  return <ul className="assistant-errors">{errors.map((entry) => <li key={`${entry.field}-${entry.code}`}>{entry.message}</li>)}</ul>;
}

function StrengthField({ resolved, value, onChange }) {
  if (resolved.kind === "unsupported") {
    return <label className="assistant-field"><span>Reasoning Effort</span><p className="assistant-hint assistant-hint-block">{resolved.hint}</p></label>;
  }
  if (resolved.kind === "enum") {
    return (
      <label className="assistant-field">
        <span>{resolved.label}</span>
        <select value={value ?? resolved.defaultValue} onChange={(event) => onChange(event.target.value)}>
          {resolved.choices.map((choice) => <option key={choice.value} value={choice.value}>{choice.label}</option>)}
        </select>
        <small>{resolved.hint}</small>
      </label>
    );
  }
  const numeric = typeof value === "number" ? value : resolved.defaultValue;
  return (
    <label className="assistant-field">
      <span>{resolved.label}<b>{numeric}</b></span>
      <input
        type="range"
        min={resolved.minimum}
        max={resolved.maximum}
        step={resolved.step}
        value={numeric}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <small>{resolved.hint}</small>
    </label>
  );
}

function ProfileRow({ profile, active, dirty, busy, canRemove, confirming, onOpen, onDuplicate, onRemove }) {
  const vendor = providerProfile(profile.settings.provider);
  // A profile missing the credential its vendor requires is the one thing worth flagging in the
  // list: it looks complete until the first message fails.
  const incomplete = Boolean(vendor?.requiresApiKey) && !profile.settings.hasApiKey;
  return (
    <div className={`assistant-lib-row ${active ? "active" : ""}`}>
      <button type="button" className="assistant-lib-open" onClick={() => onOpen(profile.id)} disabled={busy}>
        <strong>{profile.name}{dirty && <em className="assistant-lib-dirty" title="有尚未保存的修改">未保存</em>}</strong>
        <small>
          {vendor?.label || profile.settings.provider}
          {profile.settings.model ? ` · ${profile.settings.model}` : ""}
          {incomplete && <i className="assistant-lib-warn"><TriangleAlert size={10} />缺少 API Key</i>}
        </small>
      </button>
      {active && <span className="assistant-lib-live" title="对话正在使用这套配置"><Check size={11} />使用中</span>}
      <div className="assistant-lib-tools">
        <button type="button" onClick={() => onDuplicate(profile.id)} disabled={busy} title="复制这套配置" aria-label={`复制配置 ${profile.name}`}>
          <Copy size={12} />
        </button>
        {/* Two-step rather than `window.confirm`: a modal dialog cannot be shown from every host
            this panel renders in, and a configuration with a key in it is worth a second click. */}
        <button
          type="button"
          className={confirming ? "confirming" : ""}
          onClick={() => onRemove(profile.id)}
          disabled={busy || !canRemove}
          title={canRemove ? (confirming ? "再次点击确认删除" : "删除这套配置") : "至少需要保留一套配置"}
          aria-label={`删除配置 ${profile.name}`}
        >
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  );
}

export default function AssistantSettingsPanel({ store, onStoreChange, personas, personaDiagnostics, errors, onErrorsChange }) {
  // Editors are held per profile id, so switching to compare two configurations does not discard
  // what was typed into the first. A successful save drops the entry and the row re-derives from
  // the server's normalized copy.
  const [editors, setEditors] = useState({});
  const [apiKeyVisible, setApiKeyVisible] = useState(false);
  const [fetchedModels, setFetchedModels] = useState([]);
  const [manualModel, setManualModel] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);

  const profiles = store?.profiles || [];
  const activeId = store?.activeId || "";
  const active = useMemo(() => profiles.find((profile) => profile.id === activeId) || null, [activeId, profiles]);
  const stored = useMemo(() => active ? editorFromProfile(active) : null, [active]);
  const draft = (active && editors[active.id]) || stored;
  // An edit typed and then undone is not unsaved work, so the marker compares values rather than
  // merely noting that the row has an editor entry.
  const dirtyIds = useMemo(() => new Set(profiles
    .filter((profile) => editors[profile.id] && !sameEditor(editors[profile.id], editorFromProfile(profile)))
    .map((profile) => profile.id)), [editors, profiles]);
  const dirty = Boolean(active && dirtyIds.has(active.id));

  const resolvedStrength = useMemo(
    () => resolveStrengthParameter(draft?.provider, draft?.model),
    [draft?.provider, draft?.model],
  );

  const patchDraft = useCallback((changes) => {
    if (!active) return;
    setEditors((current) => ({ ...current, [active.id]: { ...(current[active.id] || editorFromProfile(active)), ...changes } }));
  }, [active]);

  // Every mutation returns the whole store, so the list and the editor redraw from one reply.
  const applyStore = useCallback((next, { clearEditorFor } = {}) => {
    onStoreChange(next);
    if (clearEditorFor) setEditors((current) => {
      const { [clearEditorFor]: discarded, ...rest } = current;
      return rest;
    });
  }, [onStoreChange]);

  const run = useCallback(async (work, { failure } = {}) => {
    if (busy) return null;
    setBusy(true);
    setNotice("");
    onErrorsChange([]);
    try {
      return await work();
    } catch (error) {
      onErrorsChange(error instanceof AssistantRequestError ? error.errors : []);
      setNotice(error.message || failure || "操作失败");
      return null;
    } finally {
      setBusy(false);
    }
  }, [busy, onErrorsChange]);

  // Selecting a profile switches the live configuration, so a list fetched from the previous
  // service must not stay on screen offering models this one may not serve.
  const openProfile = useCallback((id) => {
    setConfirmingDelete("");
    if (id === activeId) return;
    setFetchedModels([]);
    setManualModel(false);
    void run(async () => { applyStore(await activateAssistantProfile(id)); }, { failure: "无法切换配置" });
  }, [activeId, applyStore, run]);

  const addProfile = useCallback(() => {
    void run(async () => {
      setFetchedModels([]);
      setManualModel(false);
      const next = await createAssistantProfile({});
      applyStore(next);
      setNotice("已新建配置，填写服务地址与 API Key 后保存。");
    }, { failure: "无法新建配置" });
  }, [applyStore, run]);

  const copyProfile = useCallback((id) => {
    void run(async () => {
      setFetchedModels([]);
      setManualModel(false);
      applyStore(await duplicateAssistantProfile(id));
      setNotice("已复制配置，包括其中保存的 API Key。");
    }, { failure: "无法复制该配置" });
  }, [applyStore, run]);

  const removeProfile = useCallback((id) => {
    if (confirmingDelete !== id) {
      setConfirmingDelete(id);
      setNotice("再次点击删除按钮以确认删除该配置。");
      window.setTimeout(() => setConfirmingDelete((current) => current === id ? "" : current), 4000);
      return;
    }
    setConfirmingDelete("");
    void run(async () => {
      applyStore(await deleteAssistantProfile(id), { clearEditorFor: id });
      setNotice("配置已删除。");
    }, { failure: "无法删除该配置" });
  }, [applyStore, confirmingDelete, run]);

  // Resolves with the saved store, or null when validation refused it, so callers that need a
  // stored configuration (the model-list fetch) can stop instead of acting on a rejected draft.
  const persist = useCallback(async ({ thenTest, silent } = {}) => {
    if (!active || !draft) return null;
    return run(async () => {
      // `apiKey` is sent only when the user typed one; omitting it tells the control plane to keep
      // the stored secret, which the form never receives in the first place.
      const saved = await saveAssistantProfile(active.id, {
        name: draft.name,
        settings: {
          provider: draft.provider,
          baseUrl: draft.baseUrl,
          model: draft.model,
          strength: draft.strength,
          personaId: draft.personaId,
          ...(draft.apiKey ? { apiKey: draft.apiKey } : {}),
        },
      });
      applyStore(saved, { clearEditorFor: active.id });
      if (!silent) setNotice("配置已保存。");
      if (thenTest) {
        setNotice("配置已保存，正在测试连接...");
        const result = await testAssistantConnection();
        setNotice(`连接正常：${result.model}`);
      }
      return saved;
    }, { failure: "保存失败" });
  }, [active, applyStore, draft, run]);

  // Saves first, because the service the list comes from is the one recorded in the stored
  // profile; fetching against an unsaved endpoint would show models the client would not then use.
  const loadModelList = useCallback(async () => {
    if (!(await persist({ silent: true }))) return;
    await run(async () => {
      const models = await fetchAssistantModels();
      setFetchedModels(models);
      // Show the picker straight away; the user asked for a list, not a text box.
      if (models.length) setManualModel(false);
      setNotice(models.length ? `已获取 ${models.length} 个可用模型。` : "该服务未返回模型列表，请手动填写模型名称。");
    }, { failure: "无法获取模型列表" });
  }, [persist, run]);

  const changeProvider = useCallback((providerId) => {
    const profile = providerProfile(providerId);
    if (!profile) return;
    const model = profile.models[0] || "";
    // A list fetched from the previous service says nothing about this one.
    setFetchedModels([]);
    setManualModel(false);
    // Switching provider carries the previous provider's range with it otherwise: Moonshot caps at
    // 1 where DeepSeek defaults to 1.3, which would post an out-of-range value.
    patchDraft({
      provider: profile.id,
      baseUrl: profile.baseUrl,
      model,
      strength: resolveStrengthParameter(profile.id, model).defaultValue,
    });
  }, [patchDraft]);

  const changeModel = useCallback((model) => {
    if (!draft) return;
    const previous = resolveStrengthParameter(draft.provider, draft.model);
    const next = resolveStrengthParameter(draft.provider, model);
    // A model change can swap temperature for reasoning effort; keeping the old value would send a
    // number where the provider expects one of low/medium/high.
    patchDraft({ model, strength: previous.kind === next.kind && previous.parameter === next.parameter ? draft.strength : next.defaultValue });
  }, [draft, patchDraft]);

  const providerModels = providerProfile(draft?.provider)?.models || [];
  // A live list replaces the bundled suggestions once fetched; before that the bundled list is all
  // there is to offer.
  const modelOptions = fetchedModels.length ? fetchedModels : providerModels;
  const canPickModel = modelOptions.length > 0;
  const usePicker = !manualModel && canPickModel;
  // A model the user typed by hand, or one carried over from a previous provider, still has to be
  // selectable — otherwise switching to the picker would silently drop it.
  const modelPickerOptions = [...new Set([...modelOptions, draft?.model].filter(Boolean))];

  if (!store || !active || !draft) {
    return <div className="assistant-lib"><div className="assistant-lib-form"><p className="assistant-hint">正在读取配置...</p></div></div>;
  }

  return (
    <div className="assistant-lib">
      <aside className="assistant-lib-rail">
        <div className="assistant-lib-head">
          <span>配置列表<b>{profiles.length}/{MAXIMUM_PROFILES}</b></span>
          <button
            type="button"
            className="assistant-lib-add"
            onClick={addProfile}
            disabled={busy || profiles.length >= MAXIMUM_PROFILES}
            title={profiles.length >= MAXIMUM_PROFILES ? `最多保存 ${MAXIMUM_PROFILES} 套配置` : "新建一套服务配置"}
          >
            <Plus size={12} />新建
          </button>
        </div>
        <div className="assistant-lib-list">
          {profiles.map((profile) => (
            <ProfileRow
              key={profile.id}
              profile={profile}
              active={profile.id === activeId}
              dirty={dirtyIds.has(profile.id)}
              busy={busy}
              canRemove={profiles.length > 1}
              confirming={confirmingDelete === profile.id}
              onOpen={openProfile}
              onDuplicate={copyProfile}
              onRemove={removeProfile}
            />
          ))}
        </div>
        <p className="assistant-lib-foot">选中的配置即对话正在使用的配置。</p>
      </aside>

      <div className="assistant-lib-form">
        <label className="assistant-field">
          <span>配置名称{dirty && <b>未保存</b>}</span>
          <input
            spellCheck={false}
            maxLength={MAXIMUM_PROFILE_NAME}
            placeholder="例如：DeepSeek 主力"
            value={draft.name}
            onChange={(event) => patchDraft({ name: event.target.value })}
          />
          <small>仅用于区分不同的服务配置，留空会按服务商与模型自动命名。</small>
        </label>
        <label className="assistant-field">
          <span>Vendor</span>
          <select value={draft.provider} onChange={(event) => changeProvider(event.target.value)}>
            {AI_PROVIDERS.map((provider) => <option key={provider.id} value={provider.id}>{provider.label}</option>)}
          </select>
        </label>
        <label className="assistant-field">
          <span>服务地址</span>
          <input type="url" spellCheck={false} placeholder="https://api.deepseek.com/v1" value={draft.baseUrl} onChange={(event) => patchDraft({ baseUrl: event.target.value })} />
          <small>请求会发往该地址下的 /chat/completions，不会自动补全 /v1。</small>
        </label>
        <label className="assistant-field">
          <span>API Key</span>
          <div className="assistant-key-row">
            <Key size={13} />
            <input
              type={apiKeyVisible ? "text" : "password"}
              spellCheck={false}
              autoComplete="off"
              placeholder={active.settings.hasApiKey ? `已保存 ${active.settings.apiKeyHint}，留空则保持不变` : "请填写 API Key"}
              value={draft.apiKey}
              onChange={(event) => patchDraft({ apiKey: event.target.value })}
            />
            <button type="button" onClick={() => setApiKeyVisible((current) => !current)} title={apiKeyVisible ? "隐藏" : "显示"}>
              {apiKeyVisible ? <EyeOff size={13} /> : <Eye size={13} />}
            </button>
          </div>
          <small>每套配置各自保存一份密钥，仅存放在本机 state-cache 目录，不会返回给页面，也不会写入浏览器存储。</small>
        </label>
        <label className="assistant-field">
          <span>Model ID{fetchedModels.length > 0 && <b>已拉取 {fetchedModels.length} 个</b>}</span>
          <div className="assistant-model-row">
            {/* A real select rather than an input+datalist: a datalist filters its suggestions
                against what the field already contains, so once a full model id is in the box the
                popup narrows to nothing and looks like it will not open. */}
            {usePicker ? (
              <select value={draft.model} onChange={(event) => changeModel(event.target.value)}>
                {!draft.model && <option value="">请选择模型</option>}
                {modelPickerOptions.map((model) => <option key={model} value={model}>{model}</option>)}
              </select>
            ) : (
              <input spellCheck={false} placeholder={providerModels[0] || "模型名称"} value={draft.model} onChange={(event) => changeModel(event.target.value)} />
            )}
            <button type="button" onClick={() => setManualModel((current) => !current)} disabled={!canPickModel} title={usePicker ? "改为手动输入模型名称" : "从列表中选择模型"} aria-label={usePicker ? "改为手动输入模型名称" : "从列表中选择模型"}>
              {usePicker ? <Pencil size={13} /> : <List size={13} />}
            </button>
            <button type="button" onClick={() => void loadModelList()} disabled={busy} title="保存当前配置并向服务拉取可用模型列表" aria-label="拉取可用模型列表">
              {busy ? <LoaderCircle size={13} className="spin" /> : <RefreshCw size={13} />}
            </button>
          </div>
          <small>
            {fetchedModels.length
              ? "列表来自该服务的 /models 接口。"
              : providerModels.length
                ? `内置建议：${providerModels.slice(0, 3).join("、")}${providerModels.length > 3 ? " 等" : ""}。服务商会更新模型，点右侧按钮拉取当前列表。`
                : "请填写该服务支持的模型名称，或点右侧按钮拉取列表。"}
          </small>
        </label>
        <StrengthField resolved={resolvedStrength} value={draft.strength} onChange={(value) => patchDraft({ strength: value })} />
        <label className="assistant-field">
          <span>角色设定</span>
          <select value={draft.personaId} onChange={(event) => patchDraft({ personaId: event.target.value })}>
            <option value="">不使用角色设定</option>
            {personas.map((persona) => <option key={persona.id} value={persona.id}>{persona.name}</option>)}
          </select>
          <small>{personas.find((persona) => persona.id === draft.personaId)?.description || "角色决定助手的语气与行为，可在「角色」标签页中新建与编辑。"}</small>
        </label>
        {personaDiagnostics.length > 0 && <p className="assistant-hint assistant-hint-warn"><TriangleAlert size={12} />有 {personaDiagnostics.length} 个角色文件无法解析，已跳过。</p>}
        <p className="assistant-hint">知识库尚未实现，当前仅发送所选角色的系统提示词。</p>
        <ErrorList errors={errors} />
        <div className="assistant-settings-actions">
          <button type="button" onClick={() => void persist()} disabled={busy}>保存</button>
          <button type="button" className="primary" onClick={() => void persist({ thenTest: true })} disabled={busy}>
            {busy ? <LoaderCircle size={13} className="spin" /> : <Check size={13} />}保存并测试连接
          </button>
        </div>
        {notice && <p className="assistant-notice">{notice}</p>}
      </div>
    </div>
  );
}
