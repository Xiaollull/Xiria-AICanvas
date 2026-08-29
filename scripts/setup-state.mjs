import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import path from "node:path";

function cachePath(projectRoot, filename, environment) {
  return path.resolve(projectRoot, environment.XIRAI_CACHE_DIR || ".cache", filename);
}

export function getSetupMarkerPath(projectRoot, environment = process.env) {
  const configured = environment.XIRAI_SETUP_MARKER;
  if (configured) return path.resolve(projectRoot, configured);
  return cachePath(projectRoot, "setup-complete.json", environment);
}

export function readSetupMarker(projectRoot, environment = process.env) {
  const markerPath = getSetupMarkerPath(projectRoot, environment);
  if (!existsSync(markerPath)) return null;
  try {
    const marker = JSON.parse(readFileSync(markerPath, "utf8"));
    return marker?.complete === true && marker?.product === "XiriaCanvas AI" ? marker : null;
  } catch {
    return null;
  }
}

export function writeSetupMarker(projectRoot, marker, environment = process.env) {
  const markerPath = getSetupMarkerPath(projectRoot, environment);
  mkdirSync(path.dirname(markerPath), { recursive: true });
  const temporaryPath = `${markerPath}.${process.pid}.tmp`;
  writeFileSync(temporaryPath, `${JSON.stringify(marker, null, 2)}\n`, "utf8");
  renameSync(temporaryPath, markerPath);
  return markerPath;
}

export function getSetupResumePath(projectRoot, environment = process.env) {
  return cachePath(projectRoot, "setup-resume.json", environment);
}

export function readSetupResume(projectRoot, environment = process.env) {
  const resumePath = getSetupResumePath(projectRoot, environment);
  if (!existsSync(resumePath)) return null;
  try {
    const resume = JSON.parse(readFileSync(resumePath, "utf8"));
    return resume?.schema === 1 && resume?.product === "XiriaCanvas AI" ? resume : null;
  } catch {
    return null;
  }
}

export function writeSetupResume(projectRoot, resume, environment = process.env) {
  const resumePath = getSetupResumePath(projectRoot, environment);
  mkdirSync(path.dirname(resumePath), { recursive: true });
  const temporaryPath = `${resumePath}.${process.pid}.tmp`;
  writeFileSync(temporaryPath, `${JSON.stringify(resume, null, 2)}\n`, "utf8");
  renameSync(temporaryPath, resumePath);
  return resumePath;
}
