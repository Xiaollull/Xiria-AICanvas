// Test-only child. Requires explicit isolated request/state and never defaults
// to the product's state directory or starts inference.
import { readFile } from "node:fs/promises";
import { LoraCoordinator } from "./lora-coordinator.mjs";
import { galleryOwnerParticipant } from "./lora-gallery-owner.mjs";

const requestFile = process.env.P10_CRASH_REQUEST;
if (!requestFile || !process.env.P10_PYTHON || !process.env.P10_CRASH_STAGE) throw new Error("Explicit isolated crash fixture required");
const { stateDirectory, request } = JSON.parse(await readFile(requestFile, "utf8"));
const coordinator = new LoraCoordinator({ stateDirectory,
  gallery: galleryOwnerParticipant(stateDirectory, process.env.P10_PYTHON),
  admitIdle: async () => {}, validate: async () => {},
  fault: async (stage) => { if (stage === process.env.P10_CRASH_STAGE) process.exit(73); } });
await coordinator.commit(request);
throw new Error("Crash stage was not reached");
