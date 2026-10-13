import { useEffect, useRef, useState } from "react";
import { resolveLoraReferences } from "./lora-library-client.js";
import { analysisSignature, analyzeXYZGroupCopy, explainResolution, projectResolvedGroup, projectionSignature, resolveGroupMembers } from "./xyz-group-resolution.js";
import { copyXYZManagerGroup } from "./xyz-manager-context.js";

const abortLike = (error) => error?.name === "AbortError";
const failedProjection = (error) => ({ ok: false, group: null, bindings: [], issues: [{ index: -1, name: "", value: "", code: "service", message: explainResolution(error) }] });

/**
 * One library-group -> private snapshot transaction. Resolution never writes. Confirmation
 * re-verifies with the authoritative service, re-analyses the CURRENT draft and only then applies
 * one atomic edit; any failure leaves the confirmation open with the reason in place. A late
 * result, a draft-engine switch or an unmount invalidates the transaction through one token.
 */
export function useXYZGroupCopy({ engine, draftRef, attempt, onDone, resolve = resolveLoraReferences }) {
  const [state, setState] = useState(null);
  const stateRef = useRef(null), tokenRef = useRef(0), controllerRef = useRef(null), busyRef = useRef(false), engineRef = useRef(engine);
  engineRef.current = engine;
  const publish = (value) => { stateRef.current = value; setState(value); };
  const cancel = () => {
    tokenRef.current += 1; controllerRef.current?.abort(); controllerRef.current = null; busyRef.current = false;
    publish(null);
  };
  useEffect(() => () => { tokenRef.current += 1; controllerRef.current?.abort(); }, []);
  useEffect(() => { cancel(); }, [engine]);

  /** Fresh evidence for one group; resolves to null when this attempt was superseded. */
  const verify = async (source, token) => {
    const controller = new AbortController(); controllerRef.current = controller;
    try {
      const resolution = await resolveGroupMembers(engine, source, { resolve, signal: controller.signal });
      if (token !== tokenRef.current || engineRef.current !== engine) return null;
      return { projection: projectResolvedGroup(source, resolution), revision: resolution.revision };
    } catch (error) {
      if (abortLike(error) || token !== tokenRef.current) return null;
      return { projection: failedProjection(error), revision: undefined };
    }
  };

  const start = async (group) => {
    if (busyRef.current) return;
    tokenRef.current += 1; controllerRef.current?.abort();
    const token = tokenRef.current, source = structuredClone(group);
    busyRef.current = true;
    publish({ phase: "resolving", source, projection: null, revision: undefined, error: "" });
    try {
      const result = await verify(source, token);
      if (result) publish({ phase: result.projection.ok ? "ready" : "blocked", source, projection: result.projection, revision: result.revision, error: "" });
    } finally { if (token === tokenRef.current) busyRef.current = false; }
  };

  const confirm = async (mode, shownSignature) => {
    const before = stateRef.current;
    if (!before || before.phase !== "ready" || busyRef.current) return;
    tokenRef.current += 1; controllerRef.current?.abort();
    const token = tokenRef.current;
    busyRef.current = true;
    publish({ ...before, phase: "committing", error: "" });
    const hold = (phase, projection, revision, error) => publish({ phase, source: before.source, projection, revision, error });
    try {
      const result = await verify(before.source, token);
      if (!result) return;
      const { projection, revision } = result;
      if (!projection.ok || projectionSignature(projection) !== projectionSignature(before.projection)) {
        hold(projection.ok ? "ready" : "blocked", projection, revision, "资产库在确认期间发生了变化，绑定已更新；请核对后再次确认。");
        return;
      }
      const analysis = analyzeXYZGroupCopy(draftRef.current, projection.group);
      if (analysisSignature(analysis) !== shownSignature) { hold("ready", projection, revision, "XYZ 草稿在确认期间发生了变化；请核对更新后的冲突说明再次确认。"); return; }
      if (!analysis.actions[mode]) { hold("ready", projection, revision, "当前草稿已不允许这个操作；请核对说明。"); return; }
      const applied = attempt((draft) => copyXYZManagerGroup(draft, projection.group, { mode }));
      if (!applied.ok) { hold("ready", projection, revision, applied.error); return; }
      busyRef.current = false; publish(null);
      onDone?.({ group: projection.group, mode });
    } finally { if (token === tokenRef.current) busyRef.current = false; }
  };

  return { state, start, confirm, cancel };
}
