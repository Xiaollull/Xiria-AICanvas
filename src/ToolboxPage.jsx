import { Suspense, lazy, useState } from "react";
import { Activity, ArrowUpRight, Download, FileSearch, RefreshCw, ScanLine } from "lucide-react";

const ModelDownloader = lazy(() => import("./ModelDownloader"));
const ImageInfoReader = lazy(() => import("./ImageInfoReader"));

// The toolbox is a sub-navigation, not a launcher screen: opening it lands on
// the first tool directly, so reaching the downloader still costs one click
// exactly as it did when it was a top-level page.
export const TOOLBOX_TOOLS = [
  { id: "downloader", label: "模型下载器", detail: "解析链接 · 多线程续传 · 推荐模型专区", Icon: Download },
  { id: "image-info", label: "图片信息读取", detail: "读取 PNG 生成信息 · 单张或整个文件夹", Icon: FileSearch },
];

export default function ToolboxPage({ onDownloaded, onApplyImageParameters }) {
  const [activeTool, setActiveTool] = useState(TOOLBOX_TOOLS[0].id);
  const current = TOOLBOX_TOOLS.find((tool) => tool.id === activeTool) || TOOLBOX_TOOLS[0];

  return <section className="toolbox-page">
    <nav className="toolbox-rail" aria-label="工具箱">
      <header className="toolbox-rail-head">
        <div className="toolbox-rail-kicker"><span className="toolbox-status-dot" /><span>UTILITY DECK</span><code>02 TOOLS</code></div>
        <h1>工具箱</h1>
        <p>模型获取与图像诊断</p>
      </header>
      <div className="toolbox-rail-label"><span>WORKBENCH</span><i /></div>
      {TOOLBOX_TOOLS.map((tool, index) => <button
        type="button"
        key={tool.id}
        className={activeTool === tool.id ? "active" : ""}
        aria-current={activeTool === tool.id ? "page" : undefined}
        onClick={() => setActiveTool(tool.id)}
      >
        <span className="toolbox-tool-icon"><tool.Icon size={17} /></span>
        <span className="toolbox-tool-copy"><small>0{index + 1} / TOOL</small><strong>{tool.label}</strong><em>{tool.detail}</em></span>
        <ArrowUpRight className="toolbox-tool-arrow" size={14} />
      </button>)}
      <footer className="toolbox-rail-foot">
        <div><Activity size={13} /><span>WORKSPACE STATUS</span><b>READY</b></div>
        <p>工具之间独立运行，切换时保留当前工作区。</p>
        <code><ScanLine size={11} /> LOCAL CONTROL PLANE</code>
      </footer>
    </nav>
    <div className="toolbox-stage">
      <Suspense fallback={<div className="toolbox-loading"><RefreshCw className="spin" size={22} /><span>正在加载{current.label}</span></div>}>
        {activeTool === "image-info" ? <ImageInfoReader onApplyParameters={onApplyImageParameters} /> : <ModelDownloader onDownloaded={onDownloaded} />}
      </Suspense>
    </div>
  </section>;
}
