import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const toolbox = readFileSync(new URL("../src/ToolboxPage.jsx", import.meta.url), "utf8");
const downloader = readFileSync(new URL("../src/ModelDownloader.jsx", import.meta.url), "utf8");
const styles = readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");

test("toolbox persistence errors have one live announcement outside the collapsible rail", () => {
  assert.match(toolbox, /\{error && <p className="toolbox-save-error" role="alert">\{error\}<\/p>\}/);
  assert.equal((toolbox.match(/\{error\}/g) || []).length, 1, "the error message is rendered only once");
  assert.doesNotMatch(toolbox, /toolbox-rail-error/);
  assert.match(styles, /\.toolbox-save-error\s*\{[^}]*overflow-wrap:\s*anywhere/);
});

test("narrow toolbox keeps its horizontal rail and reserves stage space for an error", () => {
  assert.match(styles, /\.toolbox-stage\.has-error\s*\{\s*grid-template-rows:\s*auto\s+minmax\(0,\s*1fr\);\s*\}/);
  assert.match(styles, /\.toolbox-stage-content\s*\{[^}]*min-height:\s*0;[^}]*height:\s*100%/);
  assert.match(styles, /\.toolbox-stage-content\s*>\s*\*\s*\{\s*height:\s*100%;\s*\}/);
  assert.match(styles, /@media\s*\(max-width:\s*860px\)[\s\S]*?\.toolbox-rail\s*\{[^}]*flex-direction:\s*row;[^}]*overflow-x:\s*auto;[^}]*overflow-y:\s*hidden/);
  assert.match(styles, /\.toolbox-rail-head,\s*\.toolbox-rail-label,\s*\.toolbox-rail-foot\s*\{\s*display:\s*none;\s*\}/);
  assert.doesNotMatch(styles, /\.toolbox-save-error\s*\{[^}]*display:\s*none/);
});

test("nested downloader owns its scroll height instead of exceeding the fixed toolbox stage", () => {
  assert.match(styles, /\.model-downloader-page\s*\{\s*height:\s*100%;\s*min-height:\s*0;[^}]*\}/);
  assert.match(styles, /\.toolbox-stage-content\s*\{[^}]*min-height:\s*0;[^}]*height:\s*100%;[^}]*overflow:\s*hidden/);
  assert.match(styles, /\.toolbox-stage-content\s*>\s*\*\s*\{\s*height:\s*100%;\s*\}/);
  assert.match(styles, /\.model-downloader-page\s*\{[^}]*overflow:\s*auto/);
});

test("toolbox keeps the compact tool selector without its large title or subtitle", () => {
  assert.match(toolbox, /label:\s*"模型下载器"/);
  assert.doesNotMatch(toolbox, /<h1>工具箱<\/h1>|模型获取与图像诊断/);
});

test("downloader auxiliary actions precede links in DOM order without a page title block", () => {
  const order = ["credentials-card", "recommended-card", "link-card", "type-card", "download-progress-card", "downloader-actions"]
    .map((className) => downloader.indexOf(className));
  assert.ok(order.every((index) => index >= 0), "all workflow sections remain present");
  assert.deepEqual(order, [...order].sort((left, right) => left - right), "visual and keyboard section order stays aligned");
  assert.doesNotMatch(downloader, /<h1>模型下载器<\/h1>|粘贴模型链接，选择保存类型与目录，再开始下载/);
  assert.match(styles, /\.credentials-card\s*\{\s*grid-column:\s*1;\s*grid-row:\s*1;\s*\}/);
  assert.match(styles, /\.recommended-card\s*\{\s*grid-column:\s*2;\s*grid-row:\s*1/);
  assert.match(styles, /\.link-card\s*\{\s*grid-column:\s*1\s*\/\s*-1;\s*grid-row:\s*2;\s*\}/);
  assert.match(styles, /\.downloader-form\s*>\s*\*\s*\{\s*grid-column:\s*1;\s*grid-row:\s*auto;\s*\}/);
});
