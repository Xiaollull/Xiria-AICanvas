import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("the main navigation names the destination 画廊", async () => {
  const app = await read("src/App.jsx");
  const nav = app.slice(app.indexOf('<nav className="main-nav"'), app.indexOf("</nav>"));
  assert.match(nav, /<ImageIcon size=\{15\} \/>画廊<\/button>/);
  assert.doesNotMatch(nav, /精选画廊/);
});

test("gallery contains a persistent prompt library with complete CRUD controls", async () => {
  const [page, backend, store] = await Promise.all([
    read("src/GalleryPage.jsx"),
    read("backend/inference_server.py"),
    read("backend/gallery.py"),
  ]);
  for (const component of ["PromptLibrary", "PromptLibraryDialog"]) assert.ok(page.includes(`function ${component}`));
  assert.match(page.slice(0, page.indexOf("} from \"lucide-react\"")), /\bImagePlus,/);
  for (const field of ["标题", "正向 Prompt", "负向 Prompt", "备注（可选）"]) assert.ok(page.includes(field), `${field} is editable`);
  assert.match(page, /method: entry \? "PATCH" : "POST"/);
  assert.match(page, /method: "DELETE"/);
  for (const route of [
    '@app.get("/api/inference/gallery/prompts")',
    '@app.post("/api/inference/gallery/prompts", status_code=201)',
    '@app.patch("/api/inference/gallery/prompts/{prompt_id}")',
    '@app.delete("/api/inference/gallery/prompts/{prompt_id}")',
  ]) assert.ok(backend.includes(route), `${route} is required`);
  assert.match(store, /CREATE TABLE prompt_entries/);
  for (const owner of ["list_prompt_entries", "create_prompt_entry", "update_prompt_entry", "delete_prompt_entry"]) assert.ok(store.includes(`def ${owner}`));
});

test("a library entry can replace both text-to-image and image-to-image prompts", async () => {
  const [app, page] = await Promise.all([read("src/App.jsx"), read("src/GalleryPage.jsx")]);
  assert.match(page, /onApply\(entry, "generate"\)/);
  assert.match(page, /onApply\(entry, "image"\)/);
  const apply = app.slice(app.indexOf("const applyPromptLibraryEntry"), app.indexOf("// The popped-out tab"));
  assert.match(apply, /setImageToImage\(\(current\) => \(\{ \.\.\.current, positive: nextPositive, negative: nextNegative \}\)\)/);
  assert.match(apply, /applyAssistantPrompt\(\{ positive: nextPositive, negative: nextNegative \}, \{ allowEmpty: true, trim: false \}\)/);
  assert.match(apply, /setActivePage\("image"\)/);
  assert.match(apply, /setActivePage\("generate"\)/);
});

test("prompt library layout has desktop, light-theme, and mobile contracts", async () => {
  const css = await read("src/styles.css");
  for (const selector of [".gallery-prompt-library", ".gallery-prompt-grid", ".gallery-prompt-card", ".gallery-prompt-dialog", ".gallery-prompt-library-nav"]) assert.ok(css.includes(`${selector} {`), `${selector} is styled`);
  assert.match(css, /html\[data-theme-mode="light"\] \.gallery-prompt-card/);
  assert.match(css, /@media \(max-width: 720px\)[\s\S]*?\.gallery-prompt-grid \{ grid-template-columns: 1fr; \}/);
});
