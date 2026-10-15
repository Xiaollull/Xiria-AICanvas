import { UPDATE_VALIDATION_CLI } from "./update-validation.mjs";

// Whether the updater of the previous release can accept this one.
//
// Until 1.3.1 the updater validated an applied release inside its own process, so that release was
// checked against the *previous* release's constants. Those installations still do that once more,
// for the update that brings them the delegating updater, and they reject any backend whose
// protocol differs from the one their validator hard-codes — with a bare AssertionError, after the
// files have been replaced. The release rehearsal applies an archive with the new tree's code and
// never sees this, which is how three protocol bumps shipped that no installed updater could apply.
const PROTOCOL_IN_VALIDATOR = /const INFERENCE_PROTOCOL = (\d+);/;
const PROTOCOL_IN_BACKEND = /^INFERENCE_PROTOCOL = (\d+)\b/m;

export function previousUpdaterAccepts({ previousValidatorSource, backendSource }) {
  if (typeof previousValidatorSource !== "string" || typeof backendSource !== "string") {
    return { ok: false, reason: "The previous validator or the current backend source is missing" };
  }
  // A delegating updater runs this release's own validator, so nothing of the old one applies.
  if (previousValidatorSource.includes(UPDATE_VALIDATION_CLI)) return { ok: true, delegated: true };
  const expected = PROTOCOL_IN_VALIDATOR.exec(previousValidatorSource)?.[1];
  const actual = PROTOCOL_IN_BACKEND.exec(backendSource)?.[1];
  if (!expected || !actual) return { ok: false, reason: "Cannot read the inference protocol of the previous validator or the current backend" };
  if (expected !== actual) {
    return {
      ok: false,
      reason: `The previous release validates updates in its own process and asserts inference protocol ${expected}, `
        + `but this release's backend declares ${actual}. Every installed copy would replace its files, fail `
        + "validation and roll back. Keep the protocol unchanged in the release that ships the delegating "
        + "updater, and bump it in a later one.",
    };
  }
  return { ok: true, delegated: false };
}
