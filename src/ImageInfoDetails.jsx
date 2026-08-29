import { useEffect, useMemo, useRef, useState } from "react";
import { Boxes, Braces, Check, ChevronRight, Search, Workflow, X } from "lucide-react";
import { formatFileSize } from "./format-size";
import { CopyButton, CopyField, ModelName, ParameterModules } from "./ImageInfoFields";
import { comfyNodeRoles } from "./image-metadata";
import {
  NODE_CATEGORIES,
  buildGraphModules,
  countByCategory,
  describeChunks,
  jsonBranch,
  jsonSummary,
  matchesQuery,
} from "./metadata-explorer";

// The full record, in blocks.
//
// The summary next door answers "what made this picture" in a dozen fields. This
// dialog is where the rest of the file lives, and for a ComfyUI export the rest
// of the file is a graph of sixty-odd custom nodes. It is presented the way its
// author wrote it — one block per `{}` — because the alternative, one `<pre>` of
// 120 KB, is a payload nobody reads twice.

const LONG_VALUE_LIMIT = 220;

function FieldValue({ field, onJump }) {
  const [open, setOpen] = useState(false);
  if (field.kind === "link") {
    return <button type="button" className="metadata-field-link" onClick={() => onJump(field.target)} title={`跳转到节点 #${field.target}`}>
      <ChevronRight size={10} />{field.text}
    </button>;
  }
  if (field.kind === "value") return <b>{field.text}</b>;
  if (!field.text) return <span className="metadata-field-blank">空</span>;
  return <div className="metadata-field-text">
    <p className={open ? "open" : ""}>{open ? field.text : field.preview}</p>
    {field.truncated && <button type="button" onClick={() => setOpen(!open)}>
      {open ? "收起" : `展开全部 ${field.chars} 字`}
    </button>}
  </div>;
}

// One node, one card. The role chips are what the summary read out of this node,
// which is the only reliable answer to "why is the prompt over here" once a
// plugin has renamed every widget.
function NodeModule({ module: item, focused, onJump, register }) {
  return <article className={`metadata-node ${focused ? "focused" : ""}`} ref={(element) => register(item.id, element)}>
    <header>
      <code>#{item.id}</code>
      <div>
        <strong>{item.classType}</strong>
        {item.title ? <span>{item.title}</span> : null}
      </div>
      {item.roles.map((role) => <em key={role}>{role}</em>)}
    </header>
    {item.fields.length ? <dl className="metadata-node-fields">
      {item.fields.map((field) => <div key={field.name}>
        <dt>{field.name}</dt>
        <dd><FieldValue field={field} onJump={onJump} /></dd>
      </div>)}
    </dl> : <p className="metadata-node-empty">该节点没有参数</p>}
  </article>;
}

// The generic half. A payload that is JSON but not a node graph — the editor's
// own `workflow` chunk, a plugin's settings blob — folds per `{}` as well, and
// a branch costs one row until somebody opens it.
function JsonRow({ name, value, depth }) {
  const branch = jsonBranch(value);
  const [open, setOpen] = useState(depth === 0 && branch.size > 0 && branch.size <= 8);
  const [full, setFull] = useState(false);

  if (branch.kind === "scalar") {
    const text = typeof value === "string" ? value : JSON.stringify(value) ?? String(value);
    const long = text.length > LONG_VALUE_LIMIT;
    return <div className="metadata-json-row">
      <span className="metadata-json-key">{name}</span>
      {long
        ? <button type="button" className="metadata-json-value expandable" onClick={() => setFull(!full)}>{full ? text : `${text.slice(0, LONG_VALUE_LIMIT)}…`}</button>
        : <b className="metadata-json-value">{text}</b>}
    </div>;
  }

  return <div className="metadata-json-branch">
    <button type="button" className="metadata-json-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
      <ChevronRight size={11} className={open ? "open" : ""} />
      <span className="metadata-json-key">{name}</span>
      <i>{jsonSummary(value)}</i>
    </button>
    {open && (branch.size
      ? <div className="metadata-json-children">
        {branch.entries.map(([key, child]) => <JsonRow key={key} name={key} value={child} depth={depth + 1} />)}
      </div>
      : <p className="metadata-json-empty">空</p>)}
  </div>;
}

function ChunkBlock({ chunk }) {
  const branch = chunk.kind === "text" ? null : jsonBranch(chunk.graph || chunk.json);
  return <div className="image-info-details-chunk">
    <header>
      <code>{chunk.keyword}</code>
      <span className={`metadata-chunk-kind ${chunk.kind}`}>
        {chunk.kind === "graph" ? <><Workflow size={10} />节点图</> : chunk.kind === "json" ? <><Braces size={10} />JSON</> : <>文本</>}
      </span>
      <i>{chunk.chars.toLocaleString("en-US")} 字符{branch ? ` · ${branch.size} 个顶层区块` : ""}</i>
      <CopyButton value={chunk.text} />
    </header>
    {branch
      ? <div className="metadata-json">
        {branch.entries.map(([key, child]) => <JsonRow key={key} name={key} value={child} depth={0} />)}
      </div>
      : <pre>{chunk.text}</pre>}
  </div>;
}

export default function ImageInfoDetails({ info, onClose }) {
  const [tab, setTab] = useState("summary");
  const [category, setCategory] = useState("all");
  const [query, setQuery] = useState("");
  const [focusId, setFocusId] = useState("");
  const cardRefs = useRef(new Map());

  const chunks = useMemo(() => describeChunks(info.raw), [info]);
  const graphChunk = useMemo(() => chunks.find((chunk) => chunk.graph), [chunks]);
  const modules = useMemo(
    () => (graphChunk ? buildGraphModules(graphChunk.graph, comfyNodeRoles(graphChunk.graph)) : []),
    [graphChunk],
  );
  const counts = useMemo(() => countByCategory(modules), [modules]);
  const visible = useMemo(
    () => modules.filter((item) => (category === "all" || item.category === category) && matchesQuery(item, query)),
    [modules, category, query],
  );

  useEffect(() => {
    const onKey = (event) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Arrow keys page through a batch with this dialog open, so a filter typed
  // for one picture must not silently hide the next one's nodes.
  useEffect(() => {
    setCategory("all");
    setQuery("");
    setFocusId("");
  }, [info]);

  // Following a link has to be able to reach a card the current filter hides,
  // so the jump clears the filter before it scrolls.
  useEffect(() => {
    if (!focusId) return undefined;
    cardRefs.current.get(focusId)?.scrollIntoView({ block: "center", behavior: "smooth" });
    const timer = window.setTimeout(() => setFocusId(""), 1800);
    return () => window.clearTimeout(timer);
  }, [focusId, visible]);

  const jumpTo = (id) => {
    setCategory("all");
    setQuery("");
    setFocusId(id);
  };
  const register = (id, element) => {
    if (element) cardRefs.current.set(id, element);
    else cardRefs.current.delete(id);
  };

  const tabs = [
    { id: "summary", label: "摘要", icon: <Boxes size={12} /> },
    modules.length ? { id: "nodes", label: "工作流节点", icon: <Workflow size={12} />, count: modules.length } : null,
    chunks.length ? { id: "raw", label: "原始数据块", icon: <Braces size={12} />, count: chunks.length } : null,
  ].filter(Boolean);
  const active = tabs.some((entry) => entry.id === tab) ? tab : "summary";

  return <div className="image-info-details-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="image-info-details-dialog" role="dialog" aria-modal="true" aria-labelledby="image-info-details-title">
      <header className="image-info-details-head">
        <div>
          <span className="eyebrow">FULL METADATA RECORD</span>
          <h2 id="image-info-details-title">详细信息</h2>
          <p>{info.name || "未命名图片"} · {info.sourceLabel}{info.bytes ? ` · ${formatFileSize(info.bytes)}` : ""}</p>
        </div>
        <button type="button" aria-label="关闭详细信息" onClick={onClose}><X size={16} /></button>
      </header>

      <nav className="image-info-details-tabs">
        {tabs.map((entry) => <button
          key={entry.id}
          type="button"
          className={active === entry.id ? "on" : ""}
          aria-current={active === entry.id}
          onClick={() => setTab(entry.id)}
        >{entry.icon}{entry.label}{entry.count ? <b>{entry.count}</b> : null}</button>)}
      </nav>

      {active === "summary" && <div className="image-info-details-body summary">
        <div className="image-info-details-section wide">
          <h3>文件与来源</h3>
          <div className="image-info-details-grid">
            <div><span>文件名</span><strong title={info.name || ""}>{info.name || "未记录"}</strong></div>
            <div><span>来源</span><strong>{info.sourceLabel}</strong></div>
            <div><span>引擎</span><strong>{info.engine || "未记录"}</strong></div>
            <div><span>文件大小</span><strong>{info.bytes ? formatFileSize(info.bytes) : "未记录"}</strong></div>
            {info.nodeCount ? <div><span>工作流节点</span><strong>{info.nodeCount}</strong></div> : null}
            <div><span>数据块</span><strong>{chunks.length ? chunks.map((chunk) => chunk.keyword).join(" · ") : "无"}</strong></div>
          </div>
        </div>

        {info.status === "ok" && <>
          <div className="image-info-details-section wide">
            <h3>提示词</h3>
            <div className="image-info-details-prompts">
              <CopyField label="正向提示词" value={info.positive} empty="未记录正向提示词" />
              <CopyField label="负向提示词" value={info.negative} empty="未记录负向提示词" />
            </div>
          </div>

          {info.parameters?.length > 0 && <div className="image-info-details-section wide">
            <h3>生成参数</h3>
            <ParameterModules parameters={info.parameters} />
          </div>}

          <div className="image-info-details-section">
            <h3>底模</h3>
            <div className="image-info-details-model">
              <ModelName match={info.checkpointMatch} fallback={info.checkpoint} />
              {info.checkpointHash ? <code title={info.checkpointHash}>{info.checkpointHash}</code> : null}
            </div>
          </div>

          <div className="image-info-details-section">
            <h3>LoRA<b className="image-info-details-count">{info.loras?.length || 0}</b></h3>
            {info.loras?.length ? <div className="info-lora-list">
              {info.loras.map((item, position) => <div key={`${item.name}-${position}`}>
                <ModelName match={item.match} fallback={item.name} />
                <b>{item.weight === null || item.weight === undefined || Number.isNaN(item.weight) ? "权重未记录" : item.weight}</b>
              </div>)}
            </div> : <p className="info-section-empty">未使用 LoRA</p>}
          </div>

          {info.animaAssets && <div className="image-info-details-section">
            <h3>Anima 组件</h3>
            <div className="info-asset-row"><span>文本编码器</span><ModelName match={info.animaAssets.text_encoder.match} fallback={info.animaAssets.text_encoder.name} /></div>
            <div className="info-asset-row"><span>VAE</span><ModelName match={info.animaAssets.vae.match} fallback={info.animaAssets.vae.name} /></div>
          </div>}

          {info.flags?.length > 0 && <div className="image-info-details-section">
            <h3>功能标记</h3>
            <div className="info-flag-list">
              {info.flags.map((flag) => <span key={flag.id} className={flag.enabled ? "on" : ""}>
                {flag.enabled ? <Check size={11} /> : null}{flag.label}{flag.enabled && flag.detail ? <i>{flag.detail}</i> : null}
              </span>)}
            </div>
          </div>}
        </>}

        {info.status !== "ok" && !chunks.length && <p className="image-info-details-nothing">该图片没有可展示的元数据区块。</p>}
      </div>}

      {active === "nodes" && <div className="image-info-details-body">
        <div className="metadata-node-toolbar">
          <div className="metadata-node-filters">
            <button type="button" className={category === "all" ? "on" : ""} onClick={() => setCategory("all")}>全部<b>{modules.length}</b></button>
            {NODE_CATEGORIES.filter((entry) => counts.get(entry.id)).map((entry) => <button
              key={entry.id}
              type="button"
              className={category === entry.id ? "on" : ""}
              onClick={() => setCategory(entry.id)}
            >{entry.label}<b>{counts.get(entry.id)}</b></button>)}
          </div>
          <label className="metadata-node-search">
            <Search size={12} />
            <input value={query} spellCheck="false" placeholder="搜索节点、参数名或取值" onChange={(event) => setQuery(event.target.value)} />
            {query ? <button type="button" aria-label="清除搜索" onClick={() => setQuery("")}><X size={11} /></button> : null}
          </label>
        </div>
        {visible.length ? <div className="metadata-node-grid">
          {visible.map((item) => <NodeModule key={item.id} module={item} focused={focusId === item.id} onJump={jumpTo} register={register} />)}
        </div> : <p className="image-info-details-nothing">没有匹配的节点。</p>}
      </div>}

      {active === "raw" && <div className="image-info-details-body">
        {chunks.map((chunk) => <ChunkBlock key={chunk.keyword} chunk={chunk} />)}
      </div>}
    </section>
  </div>;
}
