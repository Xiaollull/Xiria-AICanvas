import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

// One chunk per route. Opening /update or /lora must not download the generate workspace, and the
// workspace must not carry the two pages it never renders.
const App = lazy(() => import("./App"));
const ManualUpdatePage = lazy(() => import("./ManualUpdatePage"));
const LoraManagerPage = lazy(() => import("./LoraManagerPage"));
const AiAssistantPage = lazy(() => import("./AiAssistantPage"));

// Painted only while a route chunk is in flight. It is deliberately just the backdrop the page
// already draws, so the page's own loader takes over without a visible seam.
const routeBackdrop = (className) => <main className={className}><div className="loader-grid" /></main>;

const path = window.location.pathname;

createRoot(document.getElementById("root")).render(
  <StrictMode>
    {path === "/update" || path === "/update/"
      ? <Suspense fallback={routeBackdrop("update-shell")}><ManualUpdatePage /></Suspense>
      : path === "/lora" || path === "/lora/"
        ? <Suspense fallback={<main className="lora-page-shell"><div className="lora-page-loading">正在加载 LoRA 资产管理器</div></main>}><LoraManagerPage /></Suspense>
        : path === "/assistant" || path === "/assistant/"
          ? <Suspense fallback={<main className="assistant-page-shell"><div className="assistant-page-loading">正在加载 AI 助手</div></main>}><AiAssistantPage /></Suspense>
          : <Suspense fallback={routeBackdrop("app-shell")}><App /></Suspense>}
  </StrictMode>,
);
