import { readFile } from "node:fs/promises";
import path from "node:path";

// The hand-written notes a release may carry. The Release workflow reads them beside the version
// they describe and places them above GitHub's generated compare list, so a reader meets what
// changed before the list of commits. A release without a notes file publishes as before.
export const RELEASE_NOTES_DIRECTORY = path.join(".github", "release-notes");
// The heredoc delimiter the workflow writes the notes to GITHUB_OUTPUT with. A line equal to it
// would end the value early and turn the rest of the notes into output assignments.
export const RELEASE_NOTES_DELIMITER = "XIRAI_RELEASE_NOTES";
// GitHub caps a release body at 125 000 characters; the notes stay well inside it with room for
// the generated list and a full-package notice.
export const RELEASE_NOTES_MAX_BYTES = 60 * 1024;

const STABLE_TAG = /^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/;

/** The notes for `tag`, trimmed and with LF line endings, or "" when the release has none. */
export async function readReleaseNotes(root, tag) {
  if (!STABLE_TAG.test(String(tag))) throw new Error(`Release notes are only read for stable tags; received ${tag}`);
  let text;
  try {
    text = await readFile(path.join(root, RELEASE_NOTES_DIRECTORY, `${tag}.md`), "utf8");
  } catch (error) {
    if (error.code === "ENOENT") return "";
    throw error;
  }
  text = text.replace(/\r\n?/g, "\n").trim();
  if (!text) throw new Error(`${tag}.md is empty; delete it or write the notes`);
  if (Buffer.byteLength(text, "utf8") > RELEASE_NOTES_MAX_BYTES) {
    throw new Error(`${tag}.md is larger than ${RELEASE_NOTES_MAX_BYTES} bytes`);
  }
  if (text.split("\n").some((line) => line.trim() === RELEASE_NOTES_DELIMITER)) {
    throw new Error(`${tag}.md contains a line reserved for the workflow output delimiter`);
  }
  // Notes written for one version must not be published under another.
  const heading = `# XiriaCanvas AI ${tag.slice(1)}`;
  if (text.split("\n", 1)[0].trim() !== heading) throw new Error(`${tag}.md must start with "${heading}"`);
  return text;
}
