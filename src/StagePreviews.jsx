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
 * and the Hires result while ADetailer runs. Before the first stage lands there is nothing to show
 * but the placeholder.
 */
export function StagePreviewLive({ stage, stageLabel, onEnlarge, placeholder = "正在生成，完成后显示结果" }) {
  return <>
    <div className="step-preview-empty">{stage ? null : <><ImageIcon size={28} /><span>{placeholder}</span></>}</div>
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
