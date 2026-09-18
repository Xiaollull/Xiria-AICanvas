import { Image as ImageIcon } from "lucide-react";

/**
 * A generation publishes a picture as each stage finishes -- the base sample, then Hires, then
 * ADetailer -- and both the generate page and the image-to-image page show them the same way. The
 * markup lives here once so the two cannot drift apart; the class names are the ones the stylesheet
 * and the existing tests already know.
 */

/**
 * What the panel shows while a generation runs: the most recently finished stage stays on screen
 * for the whole of the stage that follows it, so the base sample is what is shown while Hires runs
 * and the Hires result while ADetailer runs.
 *
 * Before the first stage of this run lands there is nothing of its own to show, and that is where
 * `held` comes in: the picture the previous run finished with stays up rather than the box going
 * empty the moment the next task is submitted. It is marked as the previous result, because a
 * picture with no caption in a box that says "generating" would read as this run's output.
 */
export function StagePreviewLive({ stage, stageLabel, onEnlarge, placeholder = "正在生成，完成后显示结果", held = null }) {
  return <>
    <div className="step-preview-empty">{stage || held?.url ? null : <><ImageIcon size={28} /><span>{placeholder}</span></>}</div>
    {!stage && held?.url && <div className="stage-preview-image held-preview">
      <img src={held.url} alt="上一次生成结果" />
      <span className="stage-preview-tag">上一次结果 · {placeholder}</span>
    </div>}
    {stage && <button
      type="button"
      className="stage-preview-image"
      title={`${stage.label} · 点击放大`}
      onClick={() => onEnlarge?.(stage)}
    >
      <img src={stage.url} alt={`${stage.label} 阶段结果`} />
      <span className="stage-preview-tag">{stage.label} 已完成{stageLabel ? ` · 正在进行${stageLabel}` : ""}</span>
    </button>}
  </>;
}

/** Every stage of the finished run, smallest first, each opening the shared enlargement dialog. */
export function StageStrip({ stages, onEnlarge }) {
  if (!stages?.length) return null;
  return <div className="stage-strip" aria-label="各阶段结果">
    <div className="stage-strip-head"><span>阶段结果</span><b>{stages.length} 个阶段</b></div>
    <div className="stage-strip-items">{stages.map((stage) => <button
      type="button"
      key={`${stage.index}-${stage.stage}`}
      className="stage-strip-item"
      title={`${stage.label} · ${stage.width} × ${stage.height} · 点击放大`}
      onClick={() => onEnlarge?.(stage)}
    >
      <img src={stage.url} alt={`${stage.label} 阶段结果`} />
      <span>{stage.label}</span>
    </button>)}</div>
  </div>;
}
