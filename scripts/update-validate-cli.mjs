// Post-update validation, run by the updater as a child process from inside the tree it has just
// installed (see UPDATE_VALIDATION_CLI). The process that applied the update is the previous
// release, so validating here — with this tree's own constants and checks — is what keeps a release
// from being judged by the rules of the version it replaces.
//
// Contract 1, consumed by `validateProjectTree` in every later release, so keep it stable:
//   argv[2]  the project root to validate
//   stdout   one `@@XIRAI_UPDATE_VALIDATION@@{json}` line per event, each carrying `contract: 1`:
//            {type:"progress", progress, message} …, then exactly one of
//            {type:"result", verified:true, warnings:[{check, message}]}   exit 0
//            {type:"error", kind:"program"|"environment", message}          exit 1
import path from "node:path";
import { UPDATE_VALIDATION_CONTRACT, UPDATE_VALIDATION_EVENT_PREFIX, validateUpdatedProject } from "./update-validation.mjs";

const line = (event) => `${UPDATE_VALIDATION_EVENT_PREFIX}${JSON.stringify({ contract: UPDATE_VALIDATION_CONTRACT, ...event })}\n`;
const projectRoot = path.resolve(process.argv[2] || process.cwd());
// The verdict is final: exit as soon as it is written, so no leftover keep-alive socket or grace
// timer from a check can keep the updater waiting.
const finish = (event, code) => process.stdout.write(line(event), () => process.exit(code));

try {
  const result = await validateUpdatedProject({
    projectRoot,
    environment: process.env,
    report: (event) => process.stdout.write(line({ type: "progress", progress: event.progress, message: event.message })),
  });
  finish({ type: "result", verified: result?.verified === true, warnings: result?.warnings || [] }, 0);
} catch (error) {
  finish({ type: "error", kind: error?.kind === "environment" ? "environment" : "program", message: String(error?.message || error) }, 1);
}
