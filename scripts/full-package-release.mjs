import { readFile } from "node:fs/promises";
import path from "node:path";

import { archiveUpdateInternals, NODE_DEPENDENCY_UPDATE_ERROR } from "./archive-update.mjs";

/**
 * Declaring a release that existing installations cannot apply in place.
 *
 * The in-app updater refuses any archive whose Node dependencies differ from the installed ones,
 * because replacing `node_modules` under a running Windows process is not safe. The manual has
 * always said such a version "is delivered as a full package to install fresh" — but the release
 * workflow had no way to say so, so it simply failed, and a legitimate release could not ship.
 *
 * The declaration is a tracked file naming the exact version it applies to. That shape is the
 * point:
 *
 * * it lives in the commit being released, so the decision is reviewable beside the change that
 *   forced it, and cannot be flipped after the tag is cut;
 * * it names a version rather than being a boolean, so it expires on its own — a later release
 *   that forgets to remove it is still gated normally;
 * * and it is checked against reality below, so it cannot be used to skip the in-place rehearsal
 *   for a release that would have passed it.
 *
 * That last property is what keeps this from becoming "skip the hardest test when it is
 * inconvenient". A declaration on a release that *could* have updated in place is itself an error.
 */
export const FULL_PACKAGE_DECLARATION = path.join(".github", "full-package-release");

const STABLE_VERSION = /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/;

export async function readFullPackageDeclaration(projectRoot) {
  let contents;
  try {
    contents = await readFile(path.join(projectRoot, FULL_PACKAGE_DECLARATION), "utf8");
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
  // Comments let the file carry why the release needs this, which is the part a reviewer wants.
  const declared = contents
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"));
  if (declared.length !== 1) {
    throw new Error(`${FULL_PACKAGE_DECLARATION} must name exactly one version; found ${declared.length}`);
  }
  if (!STABLE_VERSION.test(declared[0])) {
    throw new Error(`${FULL_PACKAGE_DECLARATION} must be a strict MAJOR.MINOR.PATCH version; found ${declared[0]}`);
  }
  return declared[0];
}

/** Whether *this* version is the one the declaration names. A stale declaration gates normally. */
export async function isFullPackageRelease(projectRoot, version) {
  return (await readFullPackageDeclaration(projectRoot)) === version;
}

/**
 * Whether an installed tree and a candidate package differ in the way the updater refuses.
 *
 * Asked through the updater's own predicate rather than by comparing lockfiles here, so the answer
 * cannot drift from the decision the updater will actually make on a user's machine.
 */
export async function nodeDependenciesChanged(installedRoot, packageRoot) {
  for (const root of [installedRoot, packageRoot]) {
    // `dependencyChangesRequired` reports an unreadable file as a dependency change, which is the
    // right call for an updater facing a damaged install and the wrong one here: it would let a
    // missing file justify the declaration. So readability is established first, separately.
    for (const relative of [["package-lock.json"], ["backend", "requirements.txt"]]) {
      await readFile(path.join(root, ...relative), "utf8");
    }
  }
  try {
    await archiveUpdateInternals.dependencyChangesRequired(installedRoot, packageRoot);
    return false;
  } catch (error) {
    if (String(error?.message || "").includes(NODE_DEPENDENCY_UPDATE_ERROR)) return true;
    throw error;
  }
}

/**
 * Confirm a declared full-package release is one, and reject a declaration that is not needed.
 *
 * Both directions are failures worth stopping on. An undeclared release that changes dependencies
 * would strand every existing user at the previous version with no explanation; a declared one
 * that did not change them would quietly skip the rehearsal that proves users can upgrade.
 */
export async function assertFullPackageJustified({ installedRoot, packageRoot, version }) {
  if (!(await nodeDependenciesChanged(installedRoot, packageRoot))) {
    throw new Error(
      `${FULL_PACKAGE_DECLARATION} declares ${version} a full-package release, but its Node `
      + "dependencies match the previous release, so existing installations can update in place. "
      + "Remove the declaration and let the in-place rehearsal run.",
    );
  }
}

/** The notice prepended to the published release notes, so the reader learns it before the assets. */
export const FULL_PACKAGE_NOTICE = [
  "> **This version requires a fresh installation.**",
  ">",
  "> It changes the application's Node dependencies, so the in-app updater in",
  "> Settings → About cannot apply it: replacing `node_modules` under a running process is not",
  "> safe on Windows. Download the archive below and install it as a new copy. Your models,",
  "> outputs, generated images, `.env` and installed plugins live outside the program directory",
  "> and are not affected.",
].join("\n");
