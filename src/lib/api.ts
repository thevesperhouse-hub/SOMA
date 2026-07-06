import type { CaptionConfig, DatasetImage, TrainConfig, TrainEvent } from "../types";

// Engine base URL, resolved by context:
//  - explicit override (desktop app pointed at a remote Vast engine): localStorage "soma.engineUrl"
//  - web UI served BY the engine (cloud): same origin (http(s)://<ip>:<port>)
//  - Vite dev (:1420) or Tauri app: engine on <hostname>:8765
function computeBase(): string {
  try {
    const ov = typeof localStorage !== "undefined" ? localStorage.getItem("soma.engineUrl") : null;
    if (ov) return ov.replace(/\/+$/, "");
  } catch {
    /* localStorage unavailable */
  }
  if (typeof window === "undefined" || !window.location) return "http://127.0.0.1:8765";
  const { protocol, hostname, port, host } = window.location;
  if (protocol.startsWith("http") && port !== "1420") return `${protocol}//${host}`; // served by the engine
  return `http://${hostname || "127.0.0.1"}:8765`; // Vite dev / Tauri
}

const BASE = computeBase();
const WS = BASE.replace(/^http/, "ws") + "/ws";

export async function startCaptioning(cfg: CaptionConfig): Promise<{ ok: boolean; error?: string }> {
  const r = await fetch(`${BASE}/api/caption/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  });
  return r.json();
}

export async function stopCaptioning(): Promise<void> {
  await fetch(`${BASE}/api/caption/stop`, { method: "POST" });
}

export async function captionModelStatus(): Promise<{ model_id: string; cached: boolean }> {
  try {
    const r = await fetch(`${BASE}/api/caption/model_status`);
    return await r.json();
  } catch {
    return { model_id: "", cached: false };
  }
}

export interface ModelEntry {
  name: string;
  path: string;
  folder: string;
  zimage: boolean;
}

export interface Family {
  id: string;
  label: string;
  backend: string;
  prediction: string;
  resolution: number;
  default_base: string;
  prompt_hint: string;
  params_b: number;
  quantizable: boolean;
}

export async function listFamilies(): Promise<Family[]> {
  try {
    const r = await fetch(`${BASE}/api/families`);
    return (await r.json()).families ?? [];
  } catch {
    return [];
  }
}

export interface GpuInfo {
  cuda: boolean;
  name: string;
  vram_gb: number;
}

export async function gpuInfo(): Promise<GpuInfo> {
  try {
    const r = await fetch(`${BASE}/api/gpu`);
    return await r.json();
  } catch {
    return { cuda: false, name: "", vram_gb: 0 };
  }
}

export interface GpuStats {
  ok: boolean;
  temp?: number; util?: number; fan?: number;
  mem_used?: number; mem_total?: number; clock?: number;
  power?: number; power_limit?: number; name?: string;
}

export async function gpuStats(): Promise<GpuStats> {
  try {
    const r = await fetch(`${BASE}/api/gpu/stats`);
    return await r.json();
  } catch {
    return { ok: false };
  }
}

export interface Checkpoint {
  name: string;
  path: string;
  size_mb: number;
  mtime: number;
  arch: string;
  label: string;
  base: string;
  steps: string;
  date: string;
}

export async function listCheckpoints(dir = "output"): Promise<Checkpoint[]> {
  try {
    const r = await fetch(`${BASE}/api/checkpoints?dir=${encodeURIComponent(dir)}`);
    return (await r.json()).checkpoints ?? [];
  } catch {
    return [];
  }
}

export function downloadUrl(path: string): string {
  return `${BASE}/api/download?path=${encodeURIComponent(path)}`;
}

export async function listModels(
  arch: string,
  root = ""
): Promise<{ root: string; models: ModelEntry[] }> {
  const u = `${BASE}/api/models?arch=${encodeURIComponent(arch)}&root=${encodeURIComponent(root)}`;
  try {
    const r = await fetch(u);
    return await r.json();
  } catch {
    return { root, models: [] };
  }
}

export async function datasetList(dir: string, outputDir = ""): Promise<DatasetImage[]> {
  const u = `${BASE}/api/dataset/list?dir=${encodeURIComponent(dir)}&output_dir=${encodeURIComponent(outputDir)}`;
  const r = await fetch(u);
  return (await r.json()).images ?? [];
}

export function datasetImageUrl(path: string): string {
  return `${BASE}/api/dataset/image?path=${encodeURIComponent(path)}`;
}

// small thumbnail (grid) — much lighter to decode than the full-res image
export function datasetThumbUrl(path: string, size = 384): string {
  return `${BASE}/api/dataset/thumb?path=${encodeURIComponent(path)}&size=${size}`;
}

export async function saveCaption(path: string, text: string, outputDir = ""): Promise<void> {
  await fetch(`${BASE}/api/caption/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, text, output_dir: outputDir }),
  });
}

export function connectEvents(
  onEvent: (e: TrainEvent) => void,
  onOpen?: () => void,
  onClose?: () => void
): WebSocket {
  const ws = new WebSocket(WS);
  ws.onopen = () => onOpen?.();
  ws.onclose = () => onClose?.();
  ws.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data) as TrainEvent);
    } catch {
      /* ignore */
    }
  };
  return ws;
}

export async function startTraining(cfg: TrainConfig): Promise<{ ok: boolean; error?: string }> {
  const r = await fetch(`${BASE}/api/train/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  });
  return r.json();
}

export async function stopTraining(): Promise<void> {
  await fetch(`${BASE}/api/train/stop`, { method: "POST" });
}

export async function health(): Promise<boolean> {
  try {
    const r = await fetch(`${BASE}/api/health`);
    return (await r.json()).ok === true;
  } catch {
    return false;
  }
}
