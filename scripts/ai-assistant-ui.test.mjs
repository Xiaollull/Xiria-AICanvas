import assert from "node:assert/strict";
import test from "node:test";
import path from "node:path";
import { readFile, readdir, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const projectDirectory = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const source = (name) => readFile(path.join(projectDirectory, "src", name), "utf8");

test("the composer button is the AI assistant entry point, not the old optimise stub", async () => {
  const app = await source("App.jsx");
  assert.match(app, /className="optimize"[^>]*>[^<]*<WandSparkles size=\{16\} \/>AI 助手/);
  assert.ok(!app.includes("AI 优化"), "the old label must be gone");
  assert.ok(!app.includes("正在优化..."), "the old busy label belonged to the fake timer");
  // The removed stub appended a fixed phrase after a setTimeout; nothing may still do that.
  assert.ok(!app.includes("intricate details, atmospheric depth"), "the hardcoded fake optimisation must be gone");
  assert.ok(!/const optimizePrompt = /.test(app), "the stub handler must be removed");
});

test("Ctrl/K opens the assistant and the shortcut hint still advertises it", async () => {
  const app = await source("App.jsx");
  assert.match(app, /event\.key\.toLowerCase\(\) === "k"\)\s*\{\s*event\.preventDefault\(\);\s*setAssistantOpen\(true\);/);
  assert.match(app, /<kbd>Ctrl\/⌘ K<\/kbd>/);
});

test("the assistant loads lazily and every route boundary has a default export", async () => {
  const app = await source("App.jsx");
  assert.match(app, /const AiAssistantOverlay = lazy\(\(\) => import\("\.\/AiAssistantOverlay"\)\)/);
  const main = await source("main.jsx");
  assert.match(main, /const AiAssistantPage = lazy\(\(\) => import\("\.\/AiAssistantPage"\)\)/);
  for (const name of ["AiAssistantOverlay.jsx", "AiAssistantPage.jsx", "AiAssistant.jsx"]) {
    assert.match(await source(name), /export default function/, `${name} needs a default export to be lazy-loaded`);
  }
});

test("the standalone /assistant route is reachable and matched exactly", async () => {
  const main = await source("main.jsx");
  assert.match(main, /path === "\/assistant" \|\| path === "\/assistant\/"/);
  assert.match(main, /<AiAssistantPage \/>/);
  // A route added without its own fallback would flash the workspace backdrop instead.
  assert.match(main, /assistant-page-shell/);
});

test("the floating window is non-modal: no backdrop, no scrim, nothing blocking behind it", async () => {
  const overlay = await source("AiAssistantOverlay.jsx");
  assert.match(overlay, /aria-modal="false"/);
  // Checks rendered class names rather than the word itself, which appears in the comment
  // explaining why no backdrop exists.
  assert.ok(!/className="[^"]*(?:backdrop|scrim|mask|dimmer)/i.test(overlay),
    "the overlay must not render a covering layer");
  // Closed means absent from the tree, not hidden behind CSS over the workspace.
  assert.match(overlay, /if \(!open\) return null;/);
  const styles = await readFile(path.join(projectDirectory, "src", "styles.css"), "utf8");
  const windowRule = styles.split("\n").find((line) => line.startsWith(".assistant-window {"));
  assert.ok(windowRule, "the floating window needs its own rule");
  assert.match(windowRule, /position: fixed/);
  // A full-viewport assistant layer would swallow clicks meant for the workspace underneath.
  assert.ok(!/\.assistant-[a-z-]*\s*\{[^}]*inset:\s*0/.test(styles), "no assistant rule may cover the viewport");
});

test("the window is draggable and resizable from every edge", async () => {
  const overlay = await source("AiAssistantOverlay.jsx");
  assert.match(overlay, /beginGesture\(event, "move"\)/);
  assert.match(overlay, /RESIZE_HANDLES\.map/);
  assert.match(overlay, /onPointerMove=\{continueGesture\}/);
  const styles = await readFile(path.join(projectDirectory, "src", "styles.css"), "utf8");
  for (const handle of ["n", "s", "e", "w", "ne", "nw", "se", "sw"]) {
    assert.ok(styles.includes(`.assistant-resize-${handle} {`), `handle ${handle} needs a hit area`);
  }
});

test("pop-out opens a new tab and leaves the workspace tab intact", async () => {
  const overlay = await source("AiAssistantOverlay.jsx");
  assert.match(overlay, /window\.open\("\/assistant", "_blank"/);
  assert.match(overlay, /noopener/);
  // The pop-out control sits next to the close control in the title bar.
  const bar = overlay.slice(overlay.indexOf("assistant-window-bar"), overlay.indexOf("</header>"));
  assert.ok(bar.indexOf("SquareArrowOutUpRight") < bar.indexOf("<X size"), "pop-out must precede close");
  assert.ok(!/window\.close\(\)/.test(overlay), "popping out must never close the originating page");
});

test("the API key never reaches browser storage or the page", async () => {
  const assistant = await source("AiAssistant.jsx");
  const settingsPanel = await source("AssistantSettingsPanel.jsx");
  // The form posts a key but only ever receives a masked hint back.
  assert.match(settingsPanel, /apiKeyHint/);
  for (const [name, text] of [["AiAssistant.jsx", assistant], ["AssistantSettingsPanel.jsx", settingsPanel]]) {
    assert.ok(!/localStorage[^\n]*apiKey/.test(text), `${name} must not write the key to localStorage`);
    assert.ok(!/sessionStorage/.test(text), `${name} must not use sessionStorage`);
  }
  // The settings page holds several profiles, each with its own secret, and none of them may be
  // mirrored into the browser to save a round trip.
  assert.ok(!/localStorage/.test(settingsPanel), "the settings page must persist nothing in the browser");
  const storageKeys = [...assistant.matchAll(/localStorage\.(?:get|set|remove)Item\(([^,)]+)/g)].map((match) => match[1].trim());
  // Conversations now live on disk, so the browser may hold only the pointer to the active session
  // and the legacy key that migration reads once and clears.
  assert.deepEqual([...new Set(storageKeys)].sort(), ["ASSISTANT_ACTIVE_SESSION_KEY", "LEGACY_CONVERSATION_KEY"],
    "the browser may persist only the active-session pointer, never a transcript or a secret");
  const client = await source("assistant-client.js");
  assert.ok(!/localStorage|sessionStorage/.test(client), "the transport layer must not persist anything");
  // Every provider call is proxied through the local control plane, never issued from the page.
  // Matched on the path literals themselves rather than on `fetch("…")`, so a request routed
  // through a helper — as the profile calls are — is still covered by this rule.
  const targets = [...client.matchAll(/["`](\/[^"`${]*)/g)].map((match) => match[1]);
  assert.ok(targets.length >= 10, "every route in the transport layer must be visible to this check");
  for (const target of targets) {
    assert.ok(target.startsWith("/api/assistant/"), `unexpected direct request target: ${target}`);
  }
  assert.ok(!/https?:\/\//.test(client), "no absolute URL may appear in the transport layer");
});

test("shipped personas parse and match the documented schema", async () => {
  const directory = path.join(projectDirectory, "assistant", "personas");
  const files = (await readdir(directory)).filter((name) => name.endsWith(".json"));
  assert.ok(files.length >= 1, "at least one persona ships so the feature works out of the box");
  for (const name of files) {
    const persona = JSON.parse(await readFile(path.join(directory, name), "utf8"));
    assert.equal(persona.id, name.slice(0, -".json".length), `${name} id must match its filename`);
    assert.equal(persona.schemaVersion, 1);
    assert.ok(typeof persona.systemPrompt === "string" && persona.systemPrompt.trim().length > 40, `${name} needs a real system prompt`);
    assert.ok(typeof persona.name === "string" && persona.name.trim(), `${name} needs a display name`);
    assert.deepEqual(persona.knowledge, [], `${name} must not ship knowledge before retrieval exists`);
    // The output protocol is appended by the control plane; a persona overriding it would break
    // the apply button.
    assert.ok(!persona.systemPrompt.includes("```prompt"), `${name} must not restate the output protocol`);
  }
});

test("the knowledge base is a documented placeholder, not a half-wired feature", async () => {
  const readme = await readFile(path.join(projectDirectory, "assistant", "knowledge", "README.md"), "utf8");
  assert.match(readme, /占位|尚未实现/);
  // Normalisation forces the field empty regardless of what a file says, so no build can begin
  // shipping knowledge payloads before the retrieval stage exists.
  const model = await source("assistant-persona.js");
  assert.match(model, /knowledge: \[\]/, "persona knowledge is reported empty regardless of file contents");
  assert.ok(!/knowledge:\s*(?!\[\])\S/.test(model), "nothing may normalise knowledge to anything but an empty array");
});

test("assistant assets stay out of the build output", async () => {
  const distDirectory = path.join(projectDirectory, "dist");
  try {
    await stat(distDirectory);
  } catch {
    return; // Nothing built in this checkout; the build gate covers it instead.
  }
  const walk = async (directory) => {
    const entries = await readdir(directory, { withFileTypes: true });
    const found = [];
    for (const entry of entries) {
      const target = path.join(directory, entry.name);
      if (entry.isDirectory()) found.push(...await walk(target));
      else found.push(path.relative(distDirectory, target));
    }
    return found;
  };
  const files = await walk(distDirectory);
  // Personas are read from the project root at request time; a copy in dist would be a stale fork.
  assert.ok(!files.some((file) => file.includes("personas")), "persona files must not be bundled");
  assert.ok(!files.some((file) => file.startsWith("assistant" + path.sep)), "the assistant asset folder must not be copied into dist");
});

test("chat history is persisted as server-side sessions, not in the browser", async () => {
  const assistant = await source("AiAssistant.jsx");
  // The transcript must never be written to localStorage again; only the API may hold it.
  assert.ok(!/localStorage\.setItem\([^)]*messages/i.test(assistant), "transcripts must not go to localStorage");
  assert.match(assistant, /saveAssistantSession\(sessionId, \{ messages \}\)/);
  assert.match(assistant, /messages\.some\(\(message\) => message\.streaming\)\) return/,
    "a half-streamed bubble must never be persisted");
  // Upgrading must carry the previous single conversation over rather than discarding it.
  assert.match(assistant, /takeLegacyConversation/);
  assert.match(assistant, /localStorage\.removeItem\(LEGACY_CONVERSATION_KEY\)/, "migration must consume the legacy key");

  const client = await source("assistant-client.js");
  for (const call of ["fetchAssistantSessions", "createAssistantSession", "fetchAssistantSession", "saveAssistantSession", "deleteAssistantSession"]) {
    assert.ok(client.includes(`export async function ${call}`), `${call} must exist`);
  }
  assert.ok(/encodeURIComponent\(id\)/.test(client), "session ids must be encoded into the path");
});

test("users can start a new chat and reopen a previous one", async () => {
  const assistant = await source("AiAssistant.jsx");
  assert.match(assistant, /view === "sessions"/, "the picker needs its own view");
  assert.match(assistant, /const startNewSession = /);
  assert.match(assistant, /const openSession = /);
  assert.match(assistant, /const removeSession = /);
  // A session is created on the first message, so browsing and closing leaves no empty shells.
  assert.match(assistant, /const ensureSession = /);
  assert.match(assistant, /await ensureSession\(\)/);
  assert.ok(!/useEffect\([^)]*createAssistantSession/s.test(assistant.slice(0, assistant.indexOf("const send"))),
    "a session must not be created merely by opening the assistant");

  const styles = await readFile(path.join(projectDirectory, "src", "styles.css"), "utf8");
  for (const selector of [".assistant-sessions {", ".assistant-session-row {", ".assistant-session-open {", ".assistant-session-delete {"]) {
    assert.ok(styles.includes(selector), `${selector} is missing`);
  }
});

test("the model field can be refreshed from the service instead of trusting a bundled list", async () => {
  const settingsPanel = await source("AssistantSettingsPanel.jsx");
  assert.match(settingsPanel, /const loadModelList = /);
  assert.match(settingsPanel, /fetchAssistantModels\(\)/);
  // A live list supersedes the bundled suggestions, which are only a starting point.
  assert.match(settingsPanel, /fetchedModels\.length \? fetchedModels : providerModels/);
  // A list fetched from one service must not be offered for another — including when the service
  // changes because a different saved profile was selected.
  assert.equal((settingsPanel.match(/setFetchedModels\(\[\]\)/g) || []).length, 4,
    "changing provider, selecting, creating or copying a profile each retire the fetched list");
  const client = await source("assistant-client.js");
  assert.match(client, /export async function fetchAssistantModels/);
  const styles = await readFile(path.join(projectDirectory, "src", "styles.css"), "utf8");
  assert.ok(styles.includes(".assistant-model-row {"), "the refresh control needs a layout");
});

test("the model field is a real select, not an input+datalist that cannot be opened", async () => {
  const settingsPanel = await source("AssistantSettingsPanel.jsx");
  // A datalist filters its suggestions against the field's current value, so once a full model id
  // is present the popup narrows to nothing and the control looks broken.
  assert.ok(!settingsPanel.includes("<datalist"), "datalist cannot be used to switch models");
  assert.ok(!/list="assistant-model-options"/.test(settingsPanel));
  assert.match(settingsPanel, /usePicker \? \(\s*<select value=\{draft\.model\}/, "the picker must be a select");
  // Every candidate is rendered as an option, and a hand-typed model stays selectable.
  assert.match(settingsPanel, /modelPickerOptions\.map\(\(model\) => <option key=\{model\} value=\{model\}>\{model\}<\/option>\)/);
  assert.match(settingsPanel, /\[\.\.\.new Set\(\[\.\.\.modelOptions, draft\?\.model\]\.filter\(Boolean\)\)\]/);
  // Manual entry stays reachable for a service whose list cannot be fetched.
  assert.match(settingsPanel, /setManualModel\(\(current\) => !current\)/);
  assert.match(settingsPanel, /const usePicker = !manualModel && canPickModel/);
  // A successful fetch shows the list rather than leaving the user in a text box.
  assert.match(settingsPanel, /if \(models\.length\) setManualModel\(false\)/);
  const styles = await readFile(path.join(projectDirectory, "src", "styles.css"), "utf8");
  assert.ok(styles.includes(".assistant-model-row select {"), "the select needs to fill the row");
});

test("the standalone page fills the viewport instead of a fixed centre column", async () => {
  const styles = await readFile(path.join(projectDirectory, "src", "styles.css"), "utf8");
  const pageRules = styles.split("\n").filter((line) => line.startsWith(".assistant-page"));
  assert.ok(pageRules.length, "the page needs its own rules");
  // The old layout pinned every element to min(940px, 100%) and centred it, which left most of a
  // wide screen empty. Nothing on this route may pin a fixed outer width again.
  for (const rule of pageRules) {
    assert.ok(!/width:\s*min\(\d+px/.test(rule), `the page must not pin a fixed column width: ${rule.slice(0, 60)}`);
  }
  const shell = pageRules.find((line) => line.startsWith(".assistant-page-shell {"));
  assert.match(shell, /height: 100dvh/, "the shell owns the viewport height");
  assert.ok(!/align-items: center/.test(shell), "content is no longer centred as a card");
  // Readable line length now comes from measured padding, so the scroll area still reaches both
  // edges and the scrollbar sits at the window edge rather than mid-screen.
  assert.match(styles, /\.assistant-page \.assistant-transcript \{[^}]*padding: [^;]*calc\(\(100% - \d+px\) \/ 2\)/);
  assert.match(styles, /\.assistant-page \.assistant-composer \{[^}]*calc\(\(100% - \d+px\) \/ 2\)/);
});

test("the page lists sessions in a permanent rail; the narrow window uses a tab for the same list", async () => {
  const assistant = await source("AiAssistant.jsx");
  // One list component, two hosts: the markup cannot drift between the two surfaces.
  assert.match(assistant, /function SessionList\(/);
  assert.equal((assistant.match(/<SessionList /g) || []).length, 1, "both hosts render the same element");
  assert.match(assistant, /\{isPage && \(\s*<aside className="assistant-rail">/);
  // The rail already is the history, so the page must not also offer a tab that opens it.
  assert.match(assistant, /\{!isPage && <button type="button" className=\{view === "sessions"/);
  const styles = await readFile(path.join(projectDirectory, "src", "styles.css"), "utf8");
  for (const selector of [".assistant-rail {", ".assistant-main {", ".assistant-page {"]) {
    assert.ok(styles.includes(selector), `${selector} is missing`);
  }
  assert.match(styles, /\.assistant-page \{[^}]*flex-direction: row/, "the page puts the rail beside the chat");
});

test("the floating window can be maximised and restored without losing the user's size", async () => {
  const overlay = await source("AiAssistantOverlay.jsx");
  assert.match(overlay, /const toggleMaximize = /);
  assert.match(overlay, /setRestoreRect\(rect\)/, "the pre-maximise size is remembered");
  assert.match(overlay, /restoreRect \? clampWindowRect\(restoreRect, viewport\) : defaultWindowRect\(viewport\)/,
    "a window maximised before a reload must still restore to something usable");
  // Double-clicking the bar is the same gesture, but must not fire when a title-bar button is hit.
  assert.match(overlay, /onDoubleClick=\{\(event\) => \{ if \(!event\.target\.closest\("button"\)\) toggleMaximize\(\); \}\}/);
  // Maximised still means a floating panel over a visible workspace, never a full-viewport layer.
  const geometry = await source("assistant-window.js");
  assert.match(geometry, /x: EDGE_MARGIN, y: EDGE_MARGIN/);
});

test("the composer is one box and the chrome reports which model is live", async () => {
  const assistant = await source("AiAssistant.jsx");
  // Textarea and its controls share a border; the send button is no longer detached from the field.
  assert.match(assistant, /<div className="assistant-composer-box">/);
  assert.ok(!assistant.includes("assistant-composer-row"), "the split two-row composer is gone");
  const box = assistant.slice(assistant.indexOf("assistant-composer-box"), assistant.indexOf("assistant-notice", assistant.indexOf("assistant-composer-box")));
  assert.ok(box.indexOf("<textarea") < box.indexOf("assistant-composer-actions"), "the actions sit under the field they belong to");
  // Settings stay owned here; the window bar and the page header read them through this callback
  // rather than fetching the configuration a second time.
  assert.match(assistant, /onStatusChange\?\.\(\{ ready: Boolean\(settings\), model: settings\?\.model \|\| ""/);
  assert.ok(!/onStatusChange\?\.\([^)]*apiKey/.test(assistant), "the status report must never carry the secret");
  for (const name of ["AiAssistantOverlay.jsx", "AiAssistantPage.jsx"]) {
    const chrome = await source(name);
    assert.match(chrome, /onStatusChange=\{setStatus\}/, `${name} should show the live model`);
    // Settings load asynchronously; flashing "not configured" at someone who configured it long ago
    // is worse than showing nothing for a moment.
    assert.match(chrome, /\{status\.ready &&/, `${name} must not render a verdict before settings arrive`);
  }
});

test("the settings page manages a set of saved configurations, not a single one", async () => {
  const settingsPanel = await source("AssistantSettingsPanel.jsx");
  const client = await source("assistant-client.js");
  // Choose, create, edit, copy, delete — the five things the page exists for.
  for (const call of ["fetchAssistantProfiles", "createAssistantProfile", "saveAssistantProfile", "deleteAssistantProfile", "activateAssistantProfile", "duplicateAssistantProfile"]) {
    assert.ok(client.includes(`export function ${call}`) || client.includes(`export async function ${call}`), `${call} must exist`);
  }
  assert.match(client, /encodeURIComponent\(id\)/, "profile ids must be encoded into the path");
  for (const handler of ["const openProfile = ", "const addProfile = ", "const copyProfile = ", "const removeProfile = ", "const persist = "]) {
    assert.ok(settingsPanel.includes(handler), `${handler} is missing`);
  }
  // Selecting a profile *is* activating it: the chat, the model listing and the connection probe
  // all read "the live configuration", so an editable-but-inactive profile would silently test and
  // save against a service other than the one on screen.
  assert.match(settingsPanel, /applyStore\(await activateAssistantProfile\(id\)\)/);
  assert.match(settingsPanel, /选中的配置即对话正在使用的配置/);
  // Deleting a configuration that holds a key takes two clicks, and never a blocking dialog.
  assert.match(settingsPanel, /if \(confirmingDelete !== id\)/);
  // Checks for a call rather than the word, which appears in the comment explaining why there is
  // no dialog: one cannot be shown from every host this panel renders in.
  assert.ok(!/window\.(?:confirm|alert|prompt)\(/.test(settingsPanel), "deletion must not depend on a blocking dialog");

  const styles = await readFile(path.join(projectDirectory, "src", "styles.css"), "utf8");
  // The rail-plus-editor furniture is shared with the character interface, so these class names are
  // library-generic rather than profile-specific and one rule set lays out both pages.
  for (const selector of [".assistant-lib-rail {", ".assistant-lib-list {", ".assistant-lib-row {", ".assistant-lib-open {", ".assistant-lib-tools button {", ".assistant-lib-form {"]) {
    assert.ok(styles.includes(selector), `${selector} is missing`);
  }
  const personaPanel = await source("AssistantPersonaPanel.jsx");
  for (const shared of ["assistant-lib-rail", "assistant-lib-list", "assistant-lib-row", "assistant-lib-form"]) {
    assert.ok(settingsPanel.includes(shared) && personaPanel.includes(shared), `${shared} must be shared by both library pages`);
  }
  // One component, two hosts, exactly as the session list already is: the window is 360px at its
  // narrowest, so the list stacks above the form there and sits beside it on the page.
  assert.match(styles, /\.assistant-page \.assistant-lib \{[^}]*flex-direction: row/);
  assert.match(styles, /\.assistant-lib-list \{[^}]*max-height/, "the window list must not push the form off the panel");
  const assistant = await source("AiAssistant.jsx");
  assert.equal((assistant.match(/<AssistantSettingsPanel/g) || []).length, 1, "both hosts render the same settings page");
  assert.ok(!assistant.includes("assistant-key-row"), "the settings form must live in one place only");
});

test("unsaved edits survive switching between profiles, and are marked while they last", async () => {
  const settingsPanel = await source("AssistantSettingsPanel.jsx");
  // Editors are held per profile id. Comparing two configurations is the ordinary reason to click
  // another row, and discarding what was typed for that would be indistinguishable from a bug.
  assert.match(settingsPanel, /const \[editors, setEditors\] = useState\(\{\}\)/);
  assert.match(settingsPanel, /editors\[active\.id\]\) \|\| stored/);
  // An edit typed and then undone is not unsaved work, so the marker compares values.
  assert.match(settingsPanel, /!sameEditor\(editors\[profile\.id\], editorFromProfile\(profile\)\)/);
  // A successful save drops the entry, so the row re-derives from the server's normalized copy.
  assert.match(settingsPanel, /clearEditorFor: active\.id/);
  assert.match(settingsPanel, /clearEditorFor: id/, "a deleted profile must not leave its editor behind");
});

test("the character interface is its own surface, reachable from both hosts", async () => {
  const assistant = await source("AiAssistant.jsx");
  const personaPanel = await source("AssistantPersonaPanel.jsx");
  const client = await source("assistant-client.js");

  // A view of its own, beside the chat and the service configuration rather than buried inside one.
  assert.match(assistant, /view === "personas"/);
  assert.match(assistant, /<AssistantPersonaPanel/);
  assert.equal((assistant.match(/<AssistantPersonaPanel/g) || []).length, 1, "both hosts render the same character page");
  assert.match(assistant, /setView\("personas"\)/);

  for (const call of ["createAssistantPersona", "saveAssistantPersona", "deleteAssistantPersona"]) {
    assert.ok(client.includes(`export function ${call}`), `${call} must exist`);
  }
  for (const handler of ["const addPersona = ", "const copyPersona = ", "const removePersona = ", "const persist = ", "const patchTrait = "]) {
    assert.ok(personaPanel.includes(handler), `${handler} is missing`);
  }
  // Copying is the only way to start from a shipped character, so the button may never be gated on
  // the same flag that disables editing.
  assert.match(personaPanel, /onClick=\{\(\) => onCopy\(persona\.id\)\} disabled=\{busy\}/);
  assert.match(personaPanel, /disabled=\{busy \|\| persona\.builtIn\}/, "deleting a built-in must be refused in the UI too");
  assert.match(personaPanel, /createAssistantPersona\(\{ fromId: id \}\)/);
  assert.ok(!/window\.(?:confirm|alert|prompt)\(/.test(personaPanel), "deletion must not depend on a blocking dialog");

  const styles = await readFile(path.join(projectDirectory, "src", "styles.css"), "utf8");
  for (const selector of [".assistant-persona-traits {", ".assistant-persona-preview pre {", ".assistant-persona-badge {", ".assistant-field textarea {", ".assistant-persona-columns {"]) {
    assert.ok(styles.includes(selector), `${selector} is missing`);
  }

  // The standalone page shows the character's attributes and its composed prompt side by side and
  // insets them by a fixed margin, rather than reusing the settings form's measured 860px column,
  // which left large empty gutters on a wide screen.
  assert.match(styles, /\.assistant-page \.assistant-persona-form \{[^}]*padding: 24px 30px 32px/);
  // Both rules carry the same specificity, so source order decides which wins. While the character
  // rules sat above the shared library rules, the settings form's measured 860px column overrode
  // them and the editor stayed centred with the gutters this layout exists to remove.
  assert.ok(styles.indexOf(".assistant-page .assistant-persona-form {") > styles.indexOf(".assistant-page .assistant-lib-form {"),
    "the character layout must come after the library layout it specialises");
  assert.ok(!/\.assistant-page \.assistant-persona-form \{[^}]*calc\(\(100% - \d+px\) \/ 2\)/.test(styles),
    "the character editor must not be pinned to a centred column");
  assert.match(styles, /@media \(min-width: 900px\) \{\s*\.assistant-page \.assistant-persona-columns \{[^}]*grid-template-columns: minmax\(0, 1fr\) minmax\(0, 1fr\)/);
  assert.match(personaPanel, /<div className="assistant-persona-columns">/);
  assert.equal((personaPanel.match(/<div className="assistant-persona-column">/g) || []).length, 2);
});

test("the assistant interface renders no emoji", async () => {
  // Every mark in this UI is a lucide icon. An emoji renders in the platform's own font, so it
  // sits at a different weight and colour from everything beside it and cannot follow the theme.
  const emoji = /[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}]/u;
  for (const name of ["AiAssistant.jsx", "AssistantPersonaPanel.jsx", "AssistantSettingsPanel.jsx", "AiAssistantOverlay.jsx", "AiAssistantPage.jsx", "assistant-persona.js", "assistant-profiles.js", "assistant-client.js", "ai-assistant-protocol.js", "ai-assistant-providers.js"]) {
    const text = await source(name);
    assert.ok(!emoji.test(text), `${name} must not contain an emoji`);
  }
  const personaPanel = await source("AssistantPersonaPanel.jsx");
  assert.match(personaPanel, /<UserRound size=\{12\} className="assistant-persona-mark" \/>/, "a character is marked with an icon, not a glyph the user types");
});

test("the character editor previews the exact system message it will send", async () => {
  const personaPanel = await source("AssistantPersonaPanel.jsx");
  // Through the real composer, not a second rendering that could drift from what the control plane
  // actually sends — that is the whole reason the attributes are safe to add.
  assert.match(personaPanel, /import \{ composeSystemPrompt \} from "\.\/ai-assistant-protocol\.js"/);
  assert.match(personaPanel, /composeSystemPrompt\(personaFromEditor\(draft/);
  assert.ok(!/PROMPT_PROTOCOL_INSTRUCTION|```prompt 围栏，负向/.test(personaPanel),
    "the preview must not restate the protocol; it comes from the composer");
  // Attributes are rendered from the shared vocabulary, so a new one appears in the form for free.
  assert.match(personaPanel, /PERSONA_ATTRIBUTES\.map\(\(entry\) =>/);
  // The three list-shaped fields share one editor rather than three near-identical blocks.
  assert.match(personaPanel, /function LineListField\(/);
  assert.equal((personaPanel.match(/<LineListField/g) || []).length, 3, "specialties, rules and openers use the one list editor");

  const protocol = await source("ai-assistant-protocol.js");
  assert.match(protocol, /const sections = \[\.\.\.personaSections\(persona\)\]/,
    "one place assembles the system message, and the character model supplies its sections");
  assert.match(protocol, /sections\.push\(PROMPT_PROTOCOL_INSTRUCTION\)/);
});

test("a character's attributes and openers reach the surfaces that should show them", async () => {
  const assistant = await source("AiAssistant.jsx");
  // A character with its own openers replaces the generic three; one without falls back to them.
  assert.match(assistant, /activePersona\?\.starters\?\.length \? activePersona\.starters : CHAT_STARTERS/);
  assert.match(assistant, /const activePersona = /);
  // Choosing a character writes it onto the live service profile, which is where `personaId` lives.
  assert.match(assistant, /const useCharacterForChat = /);
  assert.match(assistant, /settings: \{ provider, baseUrl, model, strength, personaId \}/);

  const settingsPanel = await source("AssistantSettingsPanel.jsx");
  assert.match(settingsPanel, /\{persona\.name\}/, "the picker names each character");
});

test("the live configuration is reported by name, so the chrome says which profile is in use", async () => {
  const assistant = await source("AiAssistant.jsx");
  assert.match(assistant, /profile: activeProfile\?\.name \|\| ""/);
  assert.ok(!/onStatusChange\?\.\([^)]*apiKey/.test(assistant), "the status report must never carry the secret");
  for (const name of ["AiAssistantOverlay.jsx", "AiAssistantPage.jsx"]) {
    assert.match(await source(name), /status\.profile \|\| status\.model/, `${name} should name the live profile`);
  }
});

test("bundled model suggestions do not name models the providers have retired", async () => {
  const providers = await source("ai-assistant-providers.js");
  const listed = [...providers.matchAll(/models:\s*Object\.freeze\(\[([^\]]*)\]\)/g)]
    .flatMap((match) => [...match[1].matchAll(/"([^"]+)"/g)].map((entry) => entry[1]));
  assert.ok(listed.length >= 20, "every provider should carry suggestions");
  // These were withdrawn by their vendors and would 404 or route to a surprise if suggested.
  for (const retired of ["deepseek-chat", "deepseek-reasoner", "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k", "kimi-k2", "qwen-plus", "qwen-max", "qwen-turbo", "glm-4-plus", "glm-4-air", "glm-4-flash"]) {
    assert.ok(!listed.includes(retired), `${retired} was retired by its provider and must not be suggested`);
  }
});

test("vendor names, model ids and strength parameters render in English", async () => {
  const { AI_PROVIDERS, resolveStrengthParameter } = await import("../src/ai-assistant-providers.js");
  const cjk = /[㐀-䶿一-鿿豈-﫿]/;
  for (const provider of AI_PROVIDERS) {
    assert.ok(!cjk.test(provider.label), `vendor "${provider.label}" must be English`);
    assert.ok(!cjk.test(provider.strength.label || ""), `${provider.id} strength label must be English`);
    for (const model of provider.models) assert.ok(!cjk.test(model), `model id "${model}" must be English`);
    for (const override of provider.overrides) {
      assert.ok(!cjk.test(override.strength.label || ""), `${provider.id} override label must be English`);
    }
  }
  // Effort tiers are shown as the exact API token, so what the user picks is what gets sent.
  for (const model of ["gpt-5.6-sol", "gpt-5.5", "gpt-5", "o4-mini"]) {
    for (const choice of resolveStrengthParameter("openai", model).choices) {
      assert.equal(choice.label, choice.value, "an effort tier must display the value it sends");
      assert.ok(!cjk.test(choice.label));
    }
  }
  // The three field labels in the settings form.
  const settingsPanel = await source("AssistantSettingsPanel.jsx");
  assert.match(settingsPanel, /<span>Vendor<\/span>/);
  assert.match(settingsPanel, /<span>Model ID/);
  assert.match(settingsPanel, /<span>Reasoning Effort<\/span>/);
  // Surrounding UI copy stays Chinese, so this is a targeted exception and not a language switch.
  assert.match(settingsPanel, /配置名称/);
  assert.match(await source("AiAssistant.jsx"), /对话/);
  assert.match(await source("AiAssistant.jsx"), /服务配置/);
});
