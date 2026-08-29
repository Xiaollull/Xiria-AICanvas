import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const readSource = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("viewer history exposes an explorer-like nested output browser", async () => {
  const [app, backend, styles] = await Promise.all([
    readSource("src/App.jsx"),
    readSource("backend/inference_server.py"),
    readSource("src/styles.css"),
  ]);

  assert.match(app, /viewerOutputRootId/);
  assert.match(app, />本次启动<\/button>/);
  assert.match(app, />输出目录<\/button>/);
  assert.match(app, />上一级<\/button>/);
  assert.match(app, /viewerFolders\.map\(\(folder\).*viewer-history-folder/s);
  assert.match(backend, /def history_directory_listing\(folder: Path \| None = None\)/);
  assert.match(backend, /"parent_id": history_folder_token\(parent\) if parent else ""/);
  assert.match(backend, /"folder_count": child_folders/);
  assert.match(styles, /\.viewer-history-folder \{/);
  assert.doesNotMatch(app, /className="viewer-folder-picker"/);
});
