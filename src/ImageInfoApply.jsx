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
// a LoRA that was never downloaded, one that shares its name with another file,
// an upscaler that is not installed, a sampler this workspace does not offer —
// so the panel stays open afterwards and lists exactly which rows those were
// and why. Closing it is the user's acknowledgement, not a timer's.
export default function ImageInfoApply({ info, context = null, onApply, onRefresh = null, onClose }) {
  const fields = useMemo(() => imageInfoApplyFields(info), [info]);
  const [target, setTarget] = useState(IMAGE_INFO_APPLY_TARGETS[0].id);
  const [selected, setSelected] = useState(() => new Set(imageInfoDefaultFields(info)));
  // A1111 never records whether Hires ran before ADetailer; the default is its own order, and the
  // user may flip it before applying (or reorder the chain afterwards on the page).
  const [chainOrder, setChainOrder] = useState("hires-first");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  // A different picture is a different set of rows; keeping the old selection
  // would apply a field the new record never mentioned. Keyed on the picture
  // rather than the record object: re-checking the model matches replaces the
  // object without changing the picture, and must not throw away what the user
  // ticked -- least of all halfway through applying it.
  const pictureKey = [info?.name, info?.bytes, info?.checkpoint, info?.positive].join("\u0000");
  useEffect(() => {
    setSelected(new Set(imageInfoDefaultFields(info)));
    setChainOrder("hires-first");
    setResult(null);
    setError("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pictureKey]);

  const toggle = (id) => setSelected((current) => {
    const next = new Set(current);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  const run = async (fieldIds) => {
    setBusy(true);
    setError("");
    try {
      // The rows were drawn from the record as it was read, which may predate a restart or an
      // install. The match that decides which engine Apply switches to is checked again now; if it
      // cannot be checked, nothing is applied rather than acting on a claim that may be stale.
      const current = onRefresh ? await onRefresh() : info;
      const plan = buildImageInfoApplyPlan(current, fieldIds, target, { engine: context?.engine || "", maximumEdge: context?.maximumEdge });
      plan.postprocessOrder = chainOrder;
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
  const missingByName = new Map((result?.missing || []).map((item) => [item.name, item]));
  const chainField = fields.find((field) => field.id === "chain");

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
    {target === "i2i" && <p className="image-info-apply-hint">写入图生图自己的提示词、采样与后处理；当前来源图保持不变。</p>}

    <ul className="image-info-apply-fields">
      {fields.map((field) => {
        const rowMissing = result ? field.missing.some((name) => missingByName.has(name)) : false;
        return <li key={field.id} className={rowMissing ? "missing" : ""}>
          <button type="button" role="checkbox" aria-checked={selected.has(field.id)} disabled={busy} onClick={() => toggle(field.id)}>
            <i>{selected.has(field.id) && <Check size={11} />}</i>
            <span>
              <strong>{field.label}{field.shared && <em>两页共用</em>}</strong>
              <small title={field.summary}>{field.summary}</small>
            </span>
          </button>
          {field.entries && <div className="image-info-apply-entries">
            {field.entries.map((entry, position) => {
              const unapplied = result ? missingByName.get(entry.name) : null;
              const flagged = unapplied || entry.missing;
              return <span
                key={`${entry.name}-${position}`}
                className={`${flagged ? "missing" : ""} ${entry.status === "ambiguous" ? "ambiguous" : ""}`}
                title={unapplied?.reason || entry.reason || (entry.label !== entry.name ? `图片记录：${entry.name}` : entry.name)}
              >
                {flagged && <AlertTriangle size={10} />}
                {entry.label}<b>{entry.weight}</b>
              </span>;
            })}
          </div>}
          {field.id === "chain" && chainField?.candidate?.order === "assumed" && <div className="image-info-apply-order" role="radiogroup" aria-label="Hires 与 ADetailer 的执行顺序">
            <span>顺序未记录：</span>
            {[["hires-first", "先 Hires 后 ADetailer"], ["ade-first", "先 ADetailer 后 Hires"]].map(([id, label]) => <button
              type="button"
              key={id}
              role="radio"
              aria-checked={chainOrder === id}
              className={chainOrder === id ? "active" : ""}
              disabled={busy}
              onClick={() => setChainOrder(id)}
            >{label}</button>)}
          </div>}
        </li>;
      })}
    </ul>

    {result && <div className={`image-info-apply-result ${result.missing.length || result.skipped.length || result.unapplied?.length ? "warn" : ""}`} role="status">
      <strong>{imageInfoApplySummary(result)} → {IMAGE_INFO_APPLY_TARGETS.find((item) => item.id === result.target)?.label}{result.targetEngine ? ` · ${result.targetEngine}` : ""}</strong>
      {result.chainApplied?.length > 0 && <small>后处理链已启用：{result.chainApplied.join(" → ")}</small>}
      {result.missing.length > 0 && <small>未应用的模型：{result.missing.map((item) => `${item.label || item.name}（${item.reason || "未找到"}）`).join("、")}</small>}
      {result.unapplied?.length > 0 && <small>未应用的后处理：{result.unapplied.join("；")}</small>}
      {result.skipped.length > 0 && <small>本工作区没有对应选项，已跳过：{result.skipped.join("、")}</small>}
      {result.notes?.length > 0 && <small>说明：{result.notes.join("；")}</small>}
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
