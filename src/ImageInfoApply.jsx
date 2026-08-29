import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Check, WandSparkles, X } from "lucide-react";
import {
  IMAGE_INFO_APPLY_TARGETS,
  buildImageInfoApplyPlan,
  imageInfoApplyAllFields,
  imageInfoApplyFields,
  imageInfoApplySummary,
  imageInfoDefaultFields,
} from "./image-info-apply";

// Applying a picture's settings to a generate page.
//
// The result is not "done": some of what a picture records cannot exist here —
// a LoRA that was never downloaded, a sampler this workspace does not offer —
// so the panel stays open afterwards and marks exactly which rows those were.
// Closing it is the user's acknowledgement, not a timer's.
export default function ImageInfoApply({ info, onApply, onClose }) {
  const fields = useMemo(() => imageInfoApplyFields(info), [info]);
  const [target, setTarget] = useState(IMAGE_INFO_APPLY_TARGETS[0].id);
  const [selected, setSelected] = useState(() => new Set(imageInfoDefaultFields(info)));
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  // A different picture is a different set of rows; keeping the old selection
  // would apply a field the new record never mentioned.
  useEffect(() => {
    setSelected(new Set(imageInfoDefaultFields(info)));
    setResult(null);
    setError("");
  }, [info]);

  const toggle = (id) => setSelected((current) => {
    const next = new Set(current);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  const run = async (fieldIds) => {
    const plan = buildImageInfoApplyPlan(info, fieldIds, target);
    setBusy(true);
    setError("");
    try {
      await onApply(plan);
      setResult(plan);
    } catch (applyError) {
      setError(applyError.message || "参数应用失败");
    } finally {
      setBusy(false);
    }
  };

  // Which rows the last apply could not carry, so the marking survives the
  // panel re-rendering and never guesses from the record alone.
  const missingNames = new Set((result?.missing || []).map((item) => item.name));

  if (!fields.length) return null;

  return <section className="image-info-apply" aria-label="应用参数">
    <header>
      <div><strong>应用参数</strong><small>把这张图片的设置写入生图页</small></div>
      <button type="button" className="image-info-apply-close" aria-label="收起应用参数" onClick={onClose}><X size={13} /></button>
    </header>

    <div className="image-info-apply-target" role="radiogroup" aria-label="应用目标">
      {IMAGE_INFO_APPLY_TARGETS.map((item) => <button
        type="button"
        key={item.id}
        role="radio"
        aria-checked={target === item.id}
        className={target === item.id ? "active" : ""}
        disabled={busy}
        onClick={() => setTarget(item.id)}
      >{item.label}</button>)}
    </div>

    <ul className="image-info-apply-fields">
      {fields.map((field) => {
        const rowMissing = result ? field.missing.some((name) => missingNames.has(name)) : false;
        return <li key={field.id} className={rowMissing ? "missing" : ""}>
          <button type="button" role="checkbox" aria-checked={selected.has(field.id)} disabled={busy} onClick={() => toggle(field.id)}>
            <i>{selected.has(field.id) && <Check size={11} />}</i>
            <span>
              <strong>{field.label}{field.shared && <em>两页共用</em>}</strong>
              <small title={field.summary}>{field.summary}</small>
            </span>
          </button>
          {field.entries && <div className="image-info-apply-entries">
            {field.entries.map((entry, position) => <span
              key={`${entry.name}-${position}`}
              className={result && missingNames.has(entry.name) ? "missing" : ""}
              title={result && missingNames.has(entry.name) ? "本地模型目录与共享目录中都没有这个模型，未挂载" : entry.name}
            >
              {result && missingNames.has(entry.name) && <AlertTriangle size={10} />}
              {entry.name.split("/").pop()}<b>{entry.weight}</b>
            </span>)}
          </div>}
        </li>;
      })}
    </ul>

    {result && <div className={`image-info-apply-result ${result.missing.length || result.skipped.length ? "warn" : ""}`} role="status">
      <strong>{imageInfoApplySummary(result)} → {IMAGE_INFO_APPLY_TARGETS.find((item) => item.id === result.target)?.label}</strong>
      {result.missing.length > 0 && <small>缺失的模型未被写入，需要先下载或放入模型目录：{result.missing.map((item) => item.name.split("/").pop()).join("、")}</small>}
      {result.skipped.length > 0 && <small>本工作区没有对应选项，已跳过：{result.skipped.join("、")}</small>}
    </div>}
    {error && <p className="image-info-apply-error"><AlertTriangle size={12} />{error}</p>}

    <footer>
      <button type="button" disabled={busy} onClick={() => void run(imageInfoApplyAllFields(info))}>全部应用 {fields.length} 项</button>
      <button type="button" className="primary" disabled={busy || !selected.size} onClick={() => void run([...selected])}>
        <WandSparkles size={13} />应用所选 {selected.size} 项
      </button>
    </footer>
  </section>;
}
