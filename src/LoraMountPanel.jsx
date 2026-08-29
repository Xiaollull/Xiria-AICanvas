import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronUp, GripVertical, ImageOff, Images, Layers3, MessageSquareText, Target, X } from "lucide-react";

import { normalizeMountedLoras } from "./lora-state.js";
import { collectingGroupId, groupIdOfMountedLora, partitionMountedByGroup, syncMountedIntoGroups } from "./lora-groups.js";
import {
  formatWeight,
  WEIGHT_MAXIMUM,
  WEIGHT_MINIMUM,
  WEIGHT_SLIDER_STEP,
  parseWeightInput,
  sliderWeight,
  steppedWeight,
  weightPrecisionLabel,
} from "./lora-weight.js";
import {
  acceptsLoraDragOver,
  establishLoraDragSession,
  isValidLoraDragSession,
  isValidLoraDropTarget,
  reorderLoraItems,
  shouldCommitLoraDrop,
  suppressLoraDragHandleKeyboard,
} from "./lora-drag-handle.js";

// The mounted queue, rendered identically by the generate page's LoRA modal and
// by the standalone /loras page. Both used to carry their own copy of this
// markup and their own drag session, which is how the two drifted: a fix to one
// list silently left the other alone.
//
// The host keeps owning persistence. This component asks for a change through
// `onUpdateLoras` / `onUpdateGroups` and never writes state itself, so the
// modal's lock-and-broadcast path and the page's epoch-guarded PUT stay exactly
// where they were.

const PREVIEW_STORAGE_KEY = "xirai-lora-mount-previews-v1";

function readPreviewPreference() {
  try {
    return window.localStorage.getItem(PREVIEW_STORAGE_KEY) === "on";
  } catch {
    // Preview visibility is a view preference; a blocked storage costs nothing.
    return false;
  }
}

function writePreviewPreference(enabled) {
  try {
    window.localStorage.setItem(PREVIEW_STORAGE_KEY, enabled ? "on" : "off");
  } catch {
    // Ignored for the same reason.
  }
}

/**
 * The weight text field holds a draft while it is focused. A controlled input
 * bound straight to the committed number cannot be edited: clearing it to type a
 * replacement immediately writes the old value back, so the caret fights every
 * keystroke and the box looks like it rejects input.
 */
function WeightField({ item, disabled, onCommit }) {
  const [draft, setDraft] = useState(null);
  const text = draft === null ? formatWeight(item.weight) : draft;

  const commit = (value) => {
    const parsed = parseWeightInput(value);
    setDraft(null);
    if (parsed.state === "valid" && parsed.weight !== item.weight) onCommit({ weight: parsed.weight, precision: 4 });
  };

  return (
    <div className="mounted-weight-field">
      <input
        className="mounted-weight-input"
        type="text"
        inputMode="decimal"
        spellCheck={false}
        aria-label={`${item.name} 权重`}
        value={text}
        disabled={disabled}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={(event) => commit(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") { event.preventDefault(); commit(event.currentTarget.value); event.currentTarget.blur(); }
          if (event.key === "Escape") { event.preventDefault(); setDraft(null); }
        }}
      />
      <span className="mounted-weight-steppers">
        <button type="button" disabled={disabled} title="权重 +0.1" aria-label={`${item.name} 权重增加 0.1`} onClick={() => onCommit(steppedWeight(item, 1))}><ChevronUp size={11} /></button>
        <button type="button" disabled={disabled} title="权重 -0.1" aria-label={`${item.name} 权重减少 0.1`} onClick={() => onCommit(steppedWeight(item, -1))}><ChevronDown size={11} /></button>
      </span>
    </div>
  );
}

function MountedPreview({ item, previewUrlFor }) {
  const [failed, setFailed] = useState(false);
  const url = failed ? "" : (previewUrlFor?.(item) || "");
  return (
    <div className="mounted-preview">
      {url
        ? <img src={url} alt="" loading="lazy" onError={() => setFailed(true)} />
        : <span className="mounted-preview-empty"><ImageOff size={13} />无图片</span>}
    </div>
  );
}

function GroupPromptBox({ group, disabled, onChange }) {
  const [open, setOpen] = useState(false);
  const length = group.presetPrompt.length;
  return (
    <div className={`mounted-group-prompt ${open ? "open" : ""}`}>
      <button type="button" className="mounted-group-prompt-toggle" aria-expanded={open} onClick={() => setOpen((current) => !current)}>
        <MessageSquareText size={12} />
        <span>预设正向提示词</span>
        {length > 0 && <b>{length}</b>}
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>
      {open && <>
        <textarea
          rows={3}
          spellCheck={false}
          disabled={disabled}
          value={group.presetPrompt}
          placeholder="该组启用时，这段提示词会加在正向提示词最前面"
          aria-label={`${group.name} 预设正向提示词`}
          onChange={(event) => onChange(event.target.value)}
        />
        <small>不会改写生成页的提示词输入框，仅在提交生成时拼接到最前面。</small>
      </>}
    </div>
  );
}

export default function LoraMountPanel({
  loras,
  groups = [],
  locked = false,
  previewUrlFor,
  onUpdateLoras,
  onUpdateGroups,
  variant = "modal",
}) {
  const [dragIndex, setDragIndex] = useState(null);
  const [dropTarget, setDropTarget] = useState(null);
  const [showPreviews, setShowPreviews] = useState(readPreviewPreference);
  const [shiftPressed, setShiftPressed] = useState(false);
  const sessionRef = useRef(null);
  const lorasRef = useRef(loras);
  lorasRef.current = loras;

  useEffect(() => {
    const down = (event) => { if (event.key === "Shift") setShiftPressed(true); };
    const up = (event) => { if (event.key === "Shift") setShiftPressed(false); };
    const blur = () => setShiftPressed(false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
    };
  }, []);

  const clearSession = useCallback(() => {
    sessionRef.current = null;
    setDragIndex(null);
    setDropTarget(null);
  }, []);

  // A lock arriving mid-gesture (a job starting, a model switching) has to end
  // the drag; the host no longer has to remember to call this itself.
  useEffect(() => {
    if (locked) clearSession();
  }, [clearSession, locked]);

  /**
   * Every mounted edit runs through here, so a change to a grouped entry always
   * reaches its group definition. Standalone mounts carry no tag and are
   * therefore invisible to the write-back by construction.
   */
  const editMounted = useCallback((updater) => {
    let next = null;
    onUpdateLoras?.((current) => {
      next = normalizeMountedLoras(updater(current));
      return next;
    });
    if (next) onUpdateGroups?.((current) => syncMountedIntoGroups(current, next));
  }, [onUpdateGroups, onUpdateLoras]);

  const patchLora = useCallback((value, changes) => {
    editMounted((current) => current.map((item) => item.value === value ? { ...item, ...changes } : item));
  }, [editMounted]);

  const togglePreviews = () => {
    setShowPreviews((current) => {
      writePreviewPreference(!current);
      return !current;
    });
  };

  const mounted = normalizeMountedLoras(loras);
  const partitioned = partitionMountedByGroup(mounted, groups);
  const collecting = collectingGroupId(groups);
  const standalone = partitioned.standalone;
  // A collector with nothing in it yet still gets a block: it is the drop zone
  // the user was just told about, and an invisible one looks like nothing worked.
  const blocks = collecting && !partitioned.blocks.some((block) => block.group.id === collecting)
    ? [...partitioned.blocks, { group: groups.find((group) => group.id === collecting), items: [] }]
    : partitioned.blocks;
  // Drag indices address the flat mounted list, not the block a row is drawn in,
  // so reordering across a group boundary stays a single well-defined move.
  const indexOf = (value) => mounted.findIndex((item) => item.value === value);

  const hasLiveDropAt = (targetIndex) => isValidLoraDragSession({ session: sessionRef.current, items: lorasRef.current, targetIndex, locked });
  const acceptsDropAt = (event, targetIndex) => acceptsLoraDragOver({ session: sessionRef.current, items: lorasRef.current, targetIndex, locked, dataTransfer: event?.dataTransfer });

  const renderRow = (item) => {
    const index = indexOf(item.value);
    const position = ((item.weight + 5) / 10) * 100;
    const grouped = Boolean(groupIdOfMountedLora(item, groups));
    return (
      <div
        className={`lora-mounted-item ${!locked && dragIndex === index && sessionRef.current?.sourceValue === item.value ? "dragging" : ""} ${!locked && dropTarget === index && hasLiveDropAt(index) ? "drop-target" : ""} ${showPreviews ? "with-preview" : ""}`}
        key={item.value}
        onDragOver={(event) => {
          if (!acceptsDropAt(event, index)) { clearSession(); return; }
          event.preventDefault();
          setDropTarget(index);
        }}
        onDragLeave={() => {
          if (!hasLiveDropAt(index)) { clearSession(); return; }
          setDropTarget((current) => current === index ? null : current);
        }}
        onDrop={(event) => {
          const session = sessionRef.current;
          if (!isValidLoraDropTarget({ session, items: lorasRef.current, targetIndex: index, locked, dataTransfer: event.dataTransfer })) { clearSession(); return; }
          event.preventDefault();
          if (shouldCommitLoraDrop({ session, items: lorasRef.current, targetIndex: index, locked, dataTransfer: event.dataTransfer })) {
            editMounted((current) => reorderLoraItems(current, session.index, index));
          }
          clearSession();
        }}
      >
        <div className="mounted-item-head">
          <button
            type="button"
            className="mounted-drag-handle"
            data-lora-drag-handle="true"
            draggable={!locked}
            disabled={locked}
            aria-label={`拖动排序：${item.name}`}
            aria-describedby="lora-mount-pointer-only"
            title="拖动排序（仅支持鼠标或触控板；触控设备不支持原生排序）"
            onKeyDown={suppressLoraDragHandleKeyboard}
            onDragStart={(event) => {
              const session = establishLoraDragSession({ event, index, sourceValue: item.value, items: lorasRef.current, locked });
              if (!session) { event.preventDefault(); clearSession(); return; }
              sessionRef.current = session;
              setDragIndex(index);
              setDropTarget(null);
            }}
            onDragEnd={clearSession}
          ><GripVertical size={14} /></button>
          <button
            type="button"
            disabled={locked}
            className={`mounted-toggle ${item.enabled !== false ? "active" : ""}`}
            role="switch"
            aria-checked={item.enabled !== false}
            title={item.enabled !== false ? "已启用" : "已停用"}
            onClick={() => patchLora(item.value, { enabled: item.enabled === false })}
          ><i /></button>
          {showPreviews && <MountedPreview item={item} previewUrlFor={previewUrlFor} />}
          <div className="mounted-item-info">
            <strong title={item.name}>{item.name}</strong>
            <small>{item.category} · {weightPrecisionLabel(item, shiftPressed)}</small>
          </div>
          <WeightField item={item} disabled={locked} onCommit={(changes) => patchLora(item.value, changes)} />
          <button
            type="button"
            className="remove"
            title={grouped ? "从该组和挂载区一并移除" : "卸载 LoRA"}
            aria-label={`卸载 ${item.name}`}
            disabled={locked}
            onClick={() => editMounted((current) => current.filter((entry) => entry.value !== item.value))}
          ><X size={14} /></button>
        </div>
        <div className="lora-weight-control">
          <span>{WEIGHT_MINIMUM}</span>
          <input
            className="lora-weight"
            type="range"
            min={WEIGHT_MINIMUM}
            max={WEIGHT_MAXIMUM}
            step={WEIGHT_SLIDER_STEP}
            value={item.weight}
            disabled={locked}
            aria-label={`${item.name} 权重滑块`}
            style={{ "--weight-start": `${Math.min(50, position)}%`, "--weight-end": `${Math.max(50, position)}%`, "--weight-color": item.enabled === false ? "#424340" : item.weight < 0 ? "var(--orange)" : "var(--lime)" }}
            onChange={(event) => patchLora(item.value, sliderWeight(item, event.target.value, shiftPressed))}
          />
          <span>{WEIGHT_MAXIMUM}</span>
        </div>
      </div>
    );
  };

  return (
    <div className={`lora-mount-panel lora-mount-${variant}`}>
      <div className="lora-mount-toolbar">
        <span id="lora-mount-pointer-only" className="lora-drag-pointer-only">排序仅支持鼠标或触控板使用左侧六点把手拖动；触控设备不支持原生排序。</span>
        <span className="lora-mount-count">已挂载 {mounted.length} / 16{blocks.length > 0 && ` · ${blocks.length} 个组合`}</span>
        <span className="lora-mount-spacer" />
        <button
          type="button"
          className={`lora-preview-toggle ${showPreviews ? "active" : ""}`}
          role="switch"
          aria-checked={showPreviews}
          title={showPreviews ? "隐藏预览图" : "显示预览图"}
          onClick={togglePreviews}
        ><Images size={13} />预览图<i /></button>
      </div>

      {!mounted.length && (
        <div className="lora-library-empty">
          <Layers3 size={30} />
          <strong>暂无挂载的 LoRA</strong>
          <p>切换到分类标签浏览并挂载，或在「组合」中启用一个 LoRA 组</p>
        </div>
      )}

      {blocks.map(({ group, items }) => (
        <section className={`lora-mount-group ${collecting === group.id ? "collecting" : ""}`} key={group.id}>
          <header>
            <span className="lora-mount-group-mark" />
            <strong title={group.name}>{group.name}</strong>
            <small>{items.length} 个 LoRA</small>
            {collecting === group.id && <em className="lora-mount-collecting"><Target size={11} />收集中</em>}
            <span className="lora-mount-spacer" />
            <button type="button" className="lora-mount-group-off" disabled={locked} title="停用该组合" onClick={() => onUpdateGroups?.((current) => current.map((entry) => entry.id === group.id ? { ...entry, enabled: false } : entry))}>停用</button>
          </header>
          <GroupPromptBox
            group={group}
            disabled={locked}
            onChange={(presetPrompt) => onUpdateGroups?.((current) => current.map((entry) => entry.id === group.id ? { ...entry, presetPrompt } : entry))}
          />
          {items.length
            ? <div className="lora-mounted-list">{items.map(renderRow)}</div>
            : <p className="lora-mount-group-waiting">去分类里挂载 LoRA，它们会加入这个组合。</p>}
        </section>
      ))}

      {standalone.length > 0 && (
        blocks.length > 0
          ? <section className="lora-mount-group lora-mount-loose">
            <header><span className="lora-mount-group-mark" /><strong>未分组</strong><small>{standalone.length} 个 LoRA</small></header>
            <div className="lora-mounted-list">{standalone.map(renderRow)}</div>
          </section>
          : <div className="lora-mounted-list">{standalone.map(renderRow)}</div>
      )}
    </div>
  );
}
