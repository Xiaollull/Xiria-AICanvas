import { spawnSync } from "node:child_process";
import { isolatedPythonEnv, runPython } from "./python.mjs";

export function parseCudaVersion(value) {
  const match = String(value || "").match(/CUDA(?:\s+UMD)?\s+Version\s*:\s*(\d+)\.(\d+)/i);
  return match ? { major: Number(match[1]), minor: Number(match[2]), text: `${match[1]}.${match[2]}` } : null;
}

export function parseCudaDriverApiVersion(value) {
  const numeric = Number(value);
  if (!Number.isInteger(numeric) || numeric < 1000) return null;
  const major = Math.floor(numeric / 1000);
  const minor = Math.floor((numeric % 1000) / 10);
  return { major, minor, text: `${major}.${minor}` };
}

// `compute_cap` is the GPU's own architecture — 7.0 for a Tesla V100, 8.6 for an RTX 3090 — and it
// decides which CUDA wheels carry kernels the card can run. It is a separate question from the
// driver's CUDA ceiling, and the two answers routinely disagree: a current driver on a V100 reports
// CUDA 13 while the card itself has been unsupported since CUDA 13 dropped Volta.
export function parseNvidiaGpuQuery(value) {
  const line = String(value || "").split(/\r?\n/).map((item) => item.trim()).find(Boolean);
  if (!line) return null;
  const fields = line.split(",").map((field) => field.trim());
  if (fields.length < 2) return { name: fields[0] || line, driverVersion: null, computeCapability: null };
  const capability = fields.length > 2 && /^\d+\.\d+$/.test(fields[fields.length - 1]) ? fields.pop() : null;
  const driverVersion = fields.pop() || null;
  return { name: fields.join(",").trim(), driverVersion, computeCapability: capability };
}

export function parseNvidiaQuery(value) {
  const details = parseNvidiaGpuQuery(value);
  return details ? { name: details.name, driverVersion: details.driverVersion } : null;
}

export function detectNvidiaSmi(executable = process.env.NVIDIA_SMI_PATH || "nvidia-smi", runner = spawnSync) {
  const options = { encoding: "utf8", windowsHide: true, timeout: 10000 };
  // `compute_cap` needs driver 510 or newer. An older driver rejects the whole query rather than
  // one field, so fall back to the pair that has always been there.
  const detailed = runner(executable, ["--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"], options);
  const query = detailed.status === 0
    ? detailed
    : runner(executable, ["--query-gpu=name,driver_version", "--format=csv,noheader"], options);
  const summary = runner(executable, [], options);
  const details = query.status === 0 ? parseNvidiaGpuQuery(query.stdout) : null;
  const summaryText = summary.status === 0 ? `${summary.stdout || ""}\n${summary.stderr || ""}` : "";
  const available = Boolean(details?.name) || /NVIDIA-SMI/i.test(summaryText) && /NVIDIA\s+(?:GeForce|RTX|Quadro|Tesla|A\d|H\d|L\d)/i.test(summaryText);
  return {
    available,
    name: details?.name || null,
    driverVersion: details?.driverVersion || summaryText.match(/(?:Driver|KMD)\s+Version\s*:\s*([\d.]+)/i)?.[1] || null,
    cuda: parseCudaVersion(summaryText),
    computeCapability: details?.computeCapability || null,
    source: available ? "nvidia-smi" : null,
  };
}

export function detectNvidiaDriverApi(python, environment = process.env) {
  const script = [
    "import ctypes,json,os",
    "name='nvcuda.dll' if os.name=='nt' else 'libcuda.so.1'",
    "cuda=ctypes.WinDLL(name) if os.name=='nt' else ctypes.CDLL(name)",
    "status=cuda.cuInit(0)",
    "count=ctypes.c_int()",
    "version=ctypes.c_int()",
    "assert status==0 and cuda.cuDeviceGetCount(ctypes.byref(count))==0 and count.value>0",
    "assert cuda.cuDriverGetVersion(ctypes.byref(version))==0",
    "device=ctypes.c_int()",
    "assert cuda.cuDeviceGet(ctypes.byref(device),0)==0",
    "buffer=ctypes.create_string_buffer(256)",
    "cuda.cuDeviceGetName(buffer,len(buffer),device)",
    // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR. Reading them here means the
    // architecture is known before any wheel is chosen, with no PyTorch installed yet.
    "major=ctypes.c_int()",
    "minor=ctypes.c_int()",
    "ok=cuda.cuDeviceGetAttribute(ctypes.byref(major),75,device)==0 and cuda.cuDeviceGetAttribute(ctypes.byref(minor),76,device)==0",
    "print(json.dumps({'count':count.value,'version':version.value,'name':buffer.value.decode('utf-8','replace'),'computeCapability':f'{major.value}.{minor.value}' if ok else None}))",
  ].join("\n");
  const result = runPython(python, ["-I", "-c", script], {
    env: isolatedPythonEnv(environment),
    stdio: "pipe",
    encoding: "utf8",
    timeout: 10000,
  });
  if (result.status !== 0) return null;
  try {
    const details = JSON.parse(result.stdout.trim());
    const cuda = parseCudaDriverApiVersion(details.version);
    return details.count > 0 && cuda ? {
      available: true,
      name: details.name || null,
      cuda,
      computeCapability: details.computeCapability || null,
      source: "driver-api",
    } : null;
  } catch {
    return null;
  }
}

export function mergeNvidiaDetection(smi, driverApi) {
  return {
    available: Boolean(driverApi?.available || smi?.available),
    name: driverApi?.name || smi?.name || null,
    driverVersion: smi?.driverVersion || null,
    cuda: driverApi?.cuda || smi?.cuda || null,
    computeCapability: driverApi?.computeCapability || smi?.computeCapability || null,
    source: driverApi?.source || smi?.source || null,
  };
}
