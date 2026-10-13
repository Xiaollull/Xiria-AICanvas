import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../src/App.jsx', import.meta.url), 'utf8');
const page = readFileSync(new URL('../src/Wd14Page.jsx', import.meta.url), 'utf8');
const style = readFileSync(new URL('../src/styles.css', import.meta.url), 'utf8');

test('tagger occupies an independent lazy main-navigation slot between i2i and gallery', () => {
  const nav = app.slice(app.indexOf('<nav className="main-nav"'), app.indexOf('</nav>'));
  assert.ok(nav.indexOf('setActivePage("image")') < nav.indexOf('setActivePage("wd14")'));
  assert.ok(nav.indexOf('setActivePage("wd14")') < nav.indexOf('setActivePage("gallery")'));
  assert.ok(nav.indexOf('setActivePage("gallery")') < nav.indexOf('setActivePage("toolbox")'));
  assert.match(app, /const Wd14Page = lazy\(\(\) => import\("\.\/Wd14Page"\)\)/);
  assert.match(app, /activePage === "wd14" \? <Suspense/);
});

test('page uses the existing accent, responsive shell and stays out of the toolbox', () => {
  assert.match(style, /\.wd14-page \{/);
  assert.match(style, /@media \(max-width: 800px\) \{ \.wd14-page/);
  assert.match(style, /\.wd14-page button\.wd14-primary \{[^}]*var\(--lime\)/);
  assert.doesNotMatch(page, /#8b7cff|#8B7CFF|ai_config\.jsonc|GGUF/);
  assert.match(page, /const API = "\/api\/inference\/wd14"/);
  assert.match(page, /request\("\/batch"/);
  assert.match(page, /useState\("ignore"\)/);
  assert.match(page, /\["general", "sensitive", "questionable", "explicit"\]/);
  assert.match(page, /"queued", "running", "publishing"/);
  assert.match(page, /正在发布 · 不可取消/);
  assert.match(page, /failure\.status === 409/);
  assert.match(page, /无法识别映射网络盘/);
  assert.doesNotMatch(page, /系统、设备、网络共享及符号链接路径不可用/);
});

test('WD14 engine choices reuse the generation segment visual language in a keyboard-selectable 2x2 group', () => {
  assert.match(page, /className="model-segment wd14-engine-grid" role="group" aria-label="模型引擎"/);
  assert.match(page, /data-engine=\{row\.id\} className=\{`model-seg \$\{engine === row\.id \? "active" : ""\}`\} aria-pressed=\{engine === row\.id\}/);
  assert.match(page, /wd: \["WD", "SmilingWolf"\], cl: \["CL", "cella110n"\]/);
  assert.match(page, /camie: \["Camie", "Camais03"\], pixai: \["PixAI", "PixAI Labs"\]/);
  assert.match(page, /onClick=\{\(\) => changeEngine\(row\.id\)\}/);
  assert.match(style, /\.wd14-page \.wd14-engine-grid \{ grid-template-columns: repeat\(2, minmax\(0, 1fr\)\); \}/);
  assert.match(style, /\.wd14-page button:not\(\.model-seg\) \{/);
  assert.match(style, /\.wd14-page button:not\(\.model-seg\):hover:not\(:disabled\)/);
  assert.match(style, /\.wd14-page button:not\(\.model-seg\):active:not\(:disabled\)/);
  assert.match(style, /\.wd14-page button:not\(\.model-seg\):disabled/);
  assert.match(style, /\.wd14-page button:focus-visible/);
  assert.match(style, /\.wd14-page \.wd14-info-button \{ width: 32px/);
  assert.doesNotMatch(page, /value=\{engine\} onChange=\{\(event\) => changeEngine/);
});

test('image-first desk omits the hero, keeps results in the primary DOM column and bounds page scrolling', () => {
  assert.doesNotMatch(page, /<h1|wd14-header|把画面读成可编辑的标签/);
  assert.ok(page.indexOf('wd14-work"') < page.indexOf('wd14-sidebar"'));
  assert.ok(page.indexOf('wd14-results"') < page.indexOf('wd14-sidebar"'));
  assert.match(style, /\.wd14-grid \{[^}]*grid-template-columns: minmax\(0, 1fr\) minmax\(350px, 390px\)/);
  assert.match(style, /\.wd14-page \{[^}]*height: calc\(100dvh - 62px\)[^}]*overflow: hidden/);
  assert.match(page, /<details className="wd14-safety"><summary>/);
  assert.match(page, /value="ignore"/);
  for (const choice of ['copy', 'append', 'prepend']) assert.match(page, new RegExp(`value="${choice}"`));
});

test('preview URLs follow the selected file and controls have complete local tracks and readable selects', () => {
  assert.match(page, /const url = URL\.createObjectURL\(file\)/);
  assert.match(page, /return \(\) => URL\.revokeObjectURL\(url\)/);
  assert.match(page, /preview\?\.file === file/);
  assert.match(style, /\.wd14-drop img \{[^}]*object-fit: contain/);
  assert.match(style, /\.wd14-card select \{[^}]*height: 44px/);
  assert.match(style, /\.wd14-slider input\[type="range"\]::-webkit-slider-runnable-track \{[^}]*var\(--wd14-progress\)/);
  assert.match(style, /\.wd14-slider input\[type="range"\]::-moz-range-progress/);
  assert.match(page, /"--wd14-progress": `\$\{value \* 100\}%`/);
  assert.match(page, /Object\.entries\(thresholds\)\.map/);
  assert.match(page, /aria-label=\{`\$\{TAGGER_CATEGORY_LABELS\[category\]\}阈值`\}/);
  assert.match(page, /navigator\.clipboard\.writeText\(item\.text\)/);
  assert.match(page, /textFile\(item\.file\.name, item\.text\)/);
});

test('single-image auto desk always has Prompt, one thumbnail history, isolated desktop scrolling and no old tabs/queue', () => {
  assert.match(page, /if \(installed && runnable && !busy\) void analyze\(id\)/);
  assert.match(page, /useSyncExternalStore\(desk\.subscribe/);
  assert.match(page, /signal \}/);
  assert.doesNotMatch(page, /wd14-tabs|wd14-queue|multiple hidden|停止剩余图片/);
  assert.match(page, /aria-label="Prompt输出"/);
  assert.match(page, /aria-label="反推结果"/);
  assert.match(page, /<ImagePreview file=\{entry\.file\}/);
  assert.match(page, /<details className="wd14-tag-details" open><summary>/);
  assert.match(style, /\.wd14-grid \{[^}]*height: 100%[^}]*grid-template-rows: minmax\(0, 1fr\)/);
  assert.match(style, /\.wd14-sidebar \{[^}]*overflow: auto/);
  assert.match(style, /\.wd14-analysis-deck \{[^}]*grid-template-columns: minmax\(0, 1fr\) minmax\(0, 1fr\)/);
});
