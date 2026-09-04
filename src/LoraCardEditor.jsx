import { useEffect, useRef, useState } from "react";
import { ImageOff, ImagePlus, RefreshCw, RotateCcw, Sparkles, Tag, X } from "lucide-react";

import LoraImagePicker from "./LoraImagePicker.jsx";
import { fromTextField, useDraftField } from "./lora-draft-field.js";
import {
  MAXIMUM_CARD_NOTE,
  MAXIMUM_CARD_PROMPT,
  MAXIMUM_CARD_TAGS,
  MAXIMUM_CARD_TITLE,
  loraCardPresentation,
  metadataPromptText,
} from "./lora-cards.js";

// Editing what one LoRA looks like everywhere it appears.
//
// Nothing here is on the generation path. The prompt is the user's own copy of
// the trigger words — kept because the looked-up ones are often nearly right and
// nowhere to fix — and it is inserted only when they ask for it, never appended
// to a request behind their back.

/**
 * Not a `<label>`: the heading row carries a button of its own, and a control
 * nested in a label steals its own click to focus the field beside it. The name
 * is attached with `aria-label` instead.
 */
function CardField({ label, hint, value, limit, rows = 0, disabled, onCommit, children }) {
  const { dirty: _dirty, ...field } = useDraftField(value, onCommit, { multiline: rows > 0 });
  const shared = { ...field, "aria-label": label, maxLength: limit, disabled, spellCheck: false };
  return <div className="lora-card-field">
    <span>{label}{children}<b>{field.value.length} / {limit}</b></span>
    {rows ? <textarea rows={rows} {...shared} /> : <input type="text" {...shared} />}
    {hint && <small>{hint}</small>}
  </div>;
}

export default function LoraCardEditor({
  engine,
  item,
  card,
  metadata,
  loading = false,
  onPatch,
  onUploadImage,
  onInsertPrompt,
  onClose,
}) {
  const dialogRef = useRef(null);
  const closeRef = useRef(onClose);
  const pickingRef = useRef(false);
  const [picking, setPicking] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  closeRef.current = onClose;
  pickingRef.current = picking;

  useEffect(() => {
    const previousFocus = document.activeElement;
    const onKeyDown = (event) => {
      if (event.key !== "Escape") return;
      // The picker sits on top and answers for itself; and Escape inside a text
      // box reverts that box rather than throwing away the whole dialog.
      if (pickingRef.current || fromTextField(event)) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      closeRef.current();
    };
    window.addEventListener("keydown", onKeyDown, true);
    const frame = window.requestAnimationFrame(() => dialogRef.current?.querySelector("input")?.focus());
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("keydown", onKeyDown, true);
      if (previousFocus instanceof HTMLElement && previousFocus.isConnected) previousFocus.focus();
    };
  }, []);

  const presentation = loraCardPresentation({ engine, item, card, metadata });
  const lookedUpPrompt = metadataPromptText(metadata);

  const applyImage = async (source) => {
    setSaving(true);
    setError("");
    try {
      onPatch({ cover: await onUploadImage(source) });
      setPicking(false);
    } catch (uploadError) {
      setError(uploadError.message);
    } finally {
      setSaving(false);
    }
  };

  return <>
    <div className="lora-card-editor-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section ref={dialogRef} className="lora-card-editor" role="dialog" aria-modal="true" aria-labelledby="lora-card-editor-title" tabIndex="-1">
        <header>
          <div>
            <span>CARD PRESENTATION</span>
            <h2 id="lora-card-editor-title">自定义 LoRA 卡片</h2>
            <p title={item.value}>{item.value}</p>
          </div>
          <button type="button" aria-label="关闭卡片编辑" onClick={onClose}><X size={19} /></button>
        </header>

        <div className="lora-card-editor-body">
          <div className="lora-card-editor-cover">
            <div className="lora-card-cover-frame">
              {presentation.coverUrl
                ? <img src={presentation.coverUrl} alt="" />
                : <span className="lora-card-cover-empty"><ImageOff size={22} />尚未设置封面</span>}
              {saving && <span className="lora-card-cover-busy"><RefreshCw className="spin" size={18} /></span>}
            </div>
            <p className="lora-card-cover-source">
              {presentation.coverSource === "custom" ? "自定义封面" : presentation.coverSource === "metadata" ? "来自来源查询的预览图" : "没有可用图片"}
            </p>
            <div className="lora-card-cover-actions">
              <button type="button" disabled={saving} onClick={() => setPicking(true)}><ImagePlus size={13} />{presentation.coverSource === "custom" ? "更换封面" : "设置封面"}</button>
              {presentation.coverSource === "custom" && (
                <button type="button" disabled={saving} title="改回来源查询下载的预览图" onClick={() => onPatch({ cover: "" })}><RotateCcw size={13} />恢复默认</button>
              )}
            </div>
            {error && <p className="lora-card-editor-error">{error}</p>}
          </div>

          <div className="lora-card-editor-fields">
            <CardField
              label="显示名称"
              hint={presentation.renamed ? `文件名 ${presentation.fileName}` : "留空则显示文件名"}
              value={card.title}
              limit={MAXIMUM_CARD_TITLE}
              onCommit={(title) => onPatch({ title })}
            />
            <CardField
              label="提示词"
              hint="卡片和悬停预览会显示这段文字；它不会被自动加入生成请求。"
              value={card.prompt}
              limit={MAXIMUM_CARD_PROMPT}
              rows={4}
              onCommit={(prompt) => onPatch({ prompt })}
            >
              {loading
                ? <em className="lora-card-field-action"><RefreshCw className="spin" size={11} />查询中</em>
                : lookedUpPrompt && lookedUpPrompt !== card.prompt
                  ? <button type="button" className="lora-card-field-action" onClick={() => onPatch({ prompt: lookedUpPrompt })}><Sparkles size={11} />填入查询到的触发词</button>
                  : null}
            </CardField>
            {onInsertPrompt && presentation.prompt && (
              <button type="button" className="lora-card-insert" onClick={() => onInsertPrompt(presentation.prompt)}>
                <Sparkles size={13} />把这段提示词追加到正向提示词
              </button>
            )}
            <CardField
              label="标签"
              hint={`用逗号分隔，最多 ${MAXIMUM_CARD_TAGS} 个`}
              value={card.tags.join(", ")}
              limit={MAXIMUM_CARD_TAGS * 26}
              onCommit={(tags) => onPatch({ tags: tags.split(/[,，]/).map((tag) => tag.trim()).filter(Boolean) })}
            ><Tag size={11} className="lora-card-field-mark" /></CardField>
            <CardField
              label="备注"
              hint="只给自己看的说明，例如推荐权重或搭配。"
              value={card.note}
              limit={MAXIMUM_CARD_NOTE}
              rows={3}
              onCommit={(note) => onPatch({ note })}
            />
          </div>
        </div>

        <footer>
          <small>卡片信息保存在 state-cache，重新解析来源不会覆盖它。</small>
          <button type="button" onClick={onClose}>完成</button>
        </footer>
      </section>
    </div>
    {picking && <LoraImagePicker title="选择封面图片" busy={saving} onPick={applyImage} onClose={() => setPicking(false)} />}
  </>;
}
