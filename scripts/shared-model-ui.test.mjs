import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const readSource = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

const loraPage = await readSource("src/LoraManagerPage.jsx");
const downloader = await readSource("src/ModelDownloader.jsx");
const settings = await readSource("src/SharedModelDirectories.jsx");
const app = await readSource("src/App.jsx");
const styles = await readSource("src/styles.css");

test("the shared category renders the folder tree instead of a flat list", () => {
  // The category itself arrives from /api/loras, so the page must branch on the
  // server's `shared` flag rather than hard-coding a fifth id.
  assert.match(loraPage, /activeCategory\?\.shared \? <div className="lora-shared-view">/);
  assert.match(loraPage, /\(activeCategory\.roots \|\| \[\]\)\.map\(\(root\) =>/);
  assert.match(loraPage, /\(root\.folders \|\| \[\]\)\.map/);

  // Keep the initial hierarchy compact; an explicit false value opens a folder.
  assert.match(loraPage, /collapsedSharedFolders\[folderKey\] !== false/);
  assert.match(loraPage, /aria-expanded=\{!collapsed\}/);
});

test("shared LoRAs use the same card as local ones so they mount identically", () => {
  // One definition, two call sites: a shared card that diverged would silently
  // lose mounting, preview, lookup or the details dialog.
  assert.equal(loraPage.match(/const renderLibraryCard = /g)?.length, 1);
  assert.equal(loraPage.match(/renderLibraryCard\(item, /g)?.length, 2);
  assert.match(loraPage, /lora-library-card lora-page-library-card/);

  // Search filters both views through the same predicate.
  assert.equal(loraPage.match(/const matchesSearch = /g)?.length, 1);
  assert.match(loraPage, /\.filter\(matchesSearch\)/);
  assert.match(loraPage, /folder\.models\.filter\(matchesSearch\)/);
});

test("the nested grid is content, not a second scrollport", () => {
  // .lora-page-library-grid scrolls on its own. Nested inside the shared tree
  // that would trap the wheel between two scroll areas.
  assert.match(styles, /\.lora-shared-view \{[^}]*overflow-y: auto/);
  assert.match(styles, /\.lora-shared-folder \.lora-page-library-grid \{[^}]*overflow: visible/);
});

test("the floating LoRA manager renders the same shared folder hierarchy", () => {
  assert.match(app, /activeLoraCategory\?\.shared \? <div className="lora-modal-shared-view">/);
  assert.match(app, /collapsedModalSharedFolders\[folderKey\] !== false/);
  assert.match(app, /lora-modal-shared-folder/);
  assert.match(app, /folder\.models\.map\(\(item\) => renderModalLoraCard/);
  assert.match(app, /const matchesLoraSearch = /);
  assert.match(styles, /\.lora-modal-shared-folder > button strong \{[^}]*font-size: 12px/);
  assert.match(styles, /\.lora-modal-shared-folder > button b \{[^}]*font: 9px "DM Mono"/);
  assert.match(styles, /\.lora-shared-folder > button strong \{[^}]*font-size: 12px/);
  assert.match(styles, /\.lora-shared-folder > button b \{[^}]*font-size: 9px/);
});

test("the settings tab owns multiple independent shared sources", () => {
  assert.match(app, /settingsTab === "models"/);
  assert.match(app, /import\("\.\/SharedModelDirectories"\)/);
  assert.match(settings, /const SHARED_ENTRY_POINTS = \[/);
  // A models root that gets classified, and a single LoRA folder.
  assert.match(settings, /kind: "auto"/);
  assert.match(settings, /kind: "loras"/);
  // The example path has to describe the machine the folders are on, which is
  // the host, not whatever computer happens to be showing the browser.
  assert.match(settings, /placeholder: hostPathExample\("modelsRoot"\)/);
  assert.match(settings, /placeholder: hostPathExample\("loraDirectory"\)/);
  assert.doesNotMatch(settings, /[A-Z]:\\\\/);
  assert.match(settings, /className="shared-source-count"/);
  assert.match(settings, /来源数量（\{roots\.length\}\/24）/);
  assert.doesNotMatch(settings, /shared-model-overview/);
  assert.match(settings, /setDraftPath\(""\)/);
  assert.doesNotMatch(downloader, /shared-paths|SHARED_ENTRY_POINTS|共享模型目录/);
});

test("registering a folder is verify-then-confirm, and removal never deletes files", () => {
  // Granting read access outside the project is a two-step action: the scan
  // report is what makes the confirm click informed.
  assert.match(settings, /\/api\/shared-paths\/inspect/);
  assert.match(settings, /const commitSharedRoot = async/);
  assert.match(settings, /确认添加/);
  assert.match(settings, /不会写入、移动或删除任何文件/);
  assert.match(settings, /不会删除磁盘文件/);

  // Every add/update/remove refreshes the active model and LoRA catalogs from
  // App, while the settings component remains independent of the downloader.
  assert.match(app, /SharedModelDirectories onChanged=\{\(\) => \{ void refreshCheckpoints\(\); void refreshLoras\(\); \}\}/);
  assert.match(settings, /onChanged\?\.\(\)/);
});

test("the scan report distinguishes detected from usable", () => {
  // Detection is wider than support; the report must not imply a ControlNet
  // folder is wired up just because it was found.
  assert.match(settings, /item\.supported \? "" : "unsupported"/);
  assert.match(settings, /暂未接入/);
  assert.match(styles, /\.shared-report-kinds > div\.unsupported \{[^}]*opacity/);
});
