"""Sidecar server: FastAPI + WebSocket. Run locally by the Tauri app.

REST :
  GET  /api/health
  POST /api/train/start   (corps = TrainConfig)
  POST /api/train/stop
WS :
  /ws  -> training event stream (replays the latest events on
          connection so a late client sees the current progress).
"""
import asyncio
import glob
import json
import os

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from captioner import CaptionJob, caption_path, clean_path
from config import CaptionConfig, CaptionSave, TrainConfig
from families import FAMILIES, get_family
from trainer import TrainingJob

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

app = FastAPI(title="SOMA Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional token auth. If SOMA_TOKEN is set (recommended when the port is public on the
# cloud), every /api call and the /ws stream must carry it (query ?token= or header
# X-Soma-Token). Empty => no auth (local/tunnel use). Lets users skip the SSH tunnel.
_TOKEN = os.environ.get("SOMA_TOKEN", "")


@app.middleware("http")
async def _auth_guard(request, call_next):
    if _TOKEN and request.url.path.startswith("/api"):
        tok = request.query_params.get("token") or request.headers.get("x-soma-token")
        if tok != _TOKEN:
            from fastapi.responses import JSONResponse

            return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


class Hub:
    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue = asyncio.Queue()
        self.clients: set[WebSocket] = set()
        self.history: list[dict] = []
        self.job: TrainingJob | None = None
        self.caption_job: CaptionJob | None = None

    def emit_threadsafe(self, event: dict):
        """Called from the training thread."""
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, event)


hub = Hub()


@app.on_event("startup")
async def _startup():
    hub.loop = asyncio.get_running_loop()
    asyncio.create_task(_broadcaster())


async def _broadcaster():
    while True:
        event = await hub.queue.get()
        hub.history.append(event)
        if len(hub.history) > 5000:
            hub.history = hub.history[-3000:]
        for ws in list(hub.clients):
            try:
                await ws.send_text(json.dumps(event))
            except Exception:
                hub.clients.discard(ws)


@app.get("/api/health")
async def health():
    training = hub.job is not None and hub.job.is_alive()
    return {"ok": True, "training": training}


@app.post("/api/train/start")
async def start(cfg: TrainConfig):
    if hub.job is not None and hub.job.is_alive():
        return {"ok": False, "error": "A training is already running"}
    hub.history.clear()
    hub.job = TrainingJob(cfg, hub.emit_threadsafe)
    hub.job.start()
    return {"ok": True}


@app.post("/api/train/stop")
async def stop():
    if hub.job is not None:
        hub.job.stop()
    return {"ok": True}


@app.post("/api/caption/start")
async def caption_start(cfg: CaptionConfig):
    if hub.caption_job is not None and hub.caption_job.is_alive():
        return {"ok": False, "error": "A captioning is already running"}
    if hub.job is not None and hub.job.is_alive():
        return {"ok": False, "error": "A training is running — wait for it to finish"}
    hub.history.clear()  # no replay of old caption events
    hub.caption_job = CaptionJob(cfg, hub.emit_threadsafe)
    hub.caption_job.start()
    return {"ok": True}


@app.post("/api/caption/stop")
async def caption_stop():
    if hub.caption_job is not None:
        hub.caption_job.stop()
    return {"ok": True}


@app.get("/api/caption/model_status")
async def caption_model_status():
    """Whether the JoyCaption model is already downloaded (else it fetches ~8 GB on
    first use). Lets the UI show a 'ready' vs 'will download' badge upfront."""
    from captioner import JOYCAPTION, model_cached

    return {"model_id": JOYCAPTION, "cached": model_cached(JOYCAPTION)}


_ZIMAGE_PAT = ("zit", "z_image", "zimage", "z-image")
_NON_SDXL_PAT = ("flux", "wan", "qwen", "sd3", "hidream", "kolors", "hunyuan", "cosmos", "ltx")


def _detect_model_root() -> str:
    """Trouve le dossier `models` d'une install ComfyUI (env var puis emplacements
    common ones). No broad glob (slow): a targeted list."""
    env = os.environ.get("SOMA_MODEL_ROOT", "")
    if env and os.path.isdir(env):
        return env
    home = os.path.expanduser("~")
    cands = [
        os.path.join(home, "ComfyUI-Installs", "ComfyUI", "ComfyUI", "models"),
        os.path.join(home, "ComfyUI", "models"),
        os.path.join(home, "Desktop", "ComfyUI", "models"),
        os.path.join(home, "Documents", "ComfyUI", "models"),
    ]
    for c in cands:
        if os.path.isdir(c):
            return c
    return ""


def _scan_models(root: str, backend: str) -> list[dict]:
    """List the models relevant to a trainer BACKEND. Z-Image = DiT in
    diffusion_models + checkpoints tagged zit/zimage. SDXL (SDXL/Pony/Illustrious/
    NoobAI) = checkpoints minus clearly non-SDXL models (flux/wan/qwen…)."""
    out = []

    def add(folder: str):
        d = os.path.join(root, folder)
        if not os.path.isdir(d):
            return
        for p in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            name = os.path.basename(p)
            low = name.lower()
            is_z = any(k in low for k in _ZIMAGE_PAT)
            if backend == "zimage":
                if is_z:  # only Z-Image DiTs (not Flux/Wan/Qwen from the same folder)
                    out.append({"name": name, "path": p, "folder": folder, "zimage": is_z})
            elif backend == "flux":
                # Flux.1-dev + Krea DiT (same arch): "flux"/"krea" but NOT
                # kontext (edit) ni flux-2/klein (autres archis)
                if (("flux" in low or "krea" in low)
                        and not any(k in low for k in ("kontext", "flux-2", "flux2", "klein"))):
                    out.append({"name": name, "path": p, "folder": folder, "zimage": False})
            elif backend == "qwen":
                # Qwen-Image DiT (base or edit) — bf16 preferred, fp8 accepted
                if "qwen" in low and "image" in low and "vae" not in low:
                    out.append({"name": name, "path": p, "folder": folder, "zimage": False})
            elif backend == "chroma":
                if "chroma" in low and "vae" not in low:
                    out.append({"name": name, "path": p, "folder": folder, "zimage": False})
            elif backend == "lumina2":
                if "lumina" in low and "vae" not in low:
                    out.append({"name": name, "path": p, "folder": folder, "zimage": False})
            else:  # backend sdxl (SDXL/Pony/Illustrious/NoobAI)
                if folder == "checkpoints" and not is_z and not any(k in low for k in _NON_SDXL_PAT):
                    out.append({"name": name, "path": p, "folder": folder, "zimage": False})

    if backend in ("zimage", "flux", "qwen", "chroma", "lumina2"):
        add("diffusion_models")
        add("checkpoints")
    else:
        add("checkpoints")
    return out


@app.get("/api/families")
async def list_families():
    """Available model families (engine-side single source)."""
    return {"families": FAMILIES}


@app.get("/api/gpu/stats")
async def gpu_stats():
    """Live GPU telemetry via nvidia-smi (temp, load, fan, VRAM, clock, power).
    Tolerant: {ok:false} if nvidia-smi is unavailable."""
    import subprocess

    q = ("temperature.gpu,utilization.gpu,fan.speed,memory.used,memory.total,"
         "clocks.sm,power.draw,power.limit,name")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip().splitlines()[0]
        v = [x.strip() for x in out.split(",")]

        def num(x):
            try:
                return float(x)
            except (ValueError, TypeError):
                return None

        return {
            "ok": True,
            "temp": num(v[0]), "util": num(v[1]), "fan": num(v[2]),
            "mem_used": num(v[3]), "mem_total": num(v[4]), "clock": num(v[5]),
            "power": num(v[6]), "power_limit": num(v[7]),
            "name": v[8] if len(v) > 8 else "",
        }
    except Exception:
        return {"ok": False}


def _read_st_metadata(path: str) -> dict:
    """Read a .safetensors' __metadata__ without loading the tensors (just
    the JSON header: 8 length bytes + JSON)."""
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(n))
        return hdr.get("__metadata__", {}) or {}
    except Exception:
        return {}


@app.get("/api/checkpoints")
async def list_checkpoints(dir: str = "output"):
    """Exported LoRAs (recursive), most recent first, with SOMA metadata (arch,
    base, steps, date) read back from the safetensors header when present."""
    dir = clean_path(dir) or "output"
    items = []
    if os.path.isdir(dir):
        paths = glob.glob(os.path.join(dir, "**", "*.safetensors"), recursive=True)
        for p in sorted(paths, key=os.path.getmtime, reverse=True)[:100]:
            try:
                md = _read_st_metadata(p)
                arch = md.get("soma_arch", "")
                items.append({
                    "name": os.path.basename(p), "path": p,
                    "size_mb": round(os.path.getsize(p) / 1e6, 1),
                    "mtime": os.path.getmtime(p),
                    "arch": arch,
                    "label": md.get("soma_label", get_family(arch)["label"] if arch else ""),
                    "base": md.get("soma_base", ""),
                    "steps": md.get("soma_steps", ""),
                    "date": md.get("soma_date", ""),
                })
            except OSError:
                continue
    return {"checkpoints": items}


@app.get("/api/download")
async def download(path: str):
    path = clean_path(path)
    if not os.path.isfile(path):
        return Response(status_code=404)
    return FileResponse(path, filename=os.path.basename(path))


@app.get("/api/gpu")
async def gpu_info():
    """Detected VRAM (for a smart precision default). Tolerant if
    torch/CUDA is missing (demo mode, base install)."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {"cuda": True, "name": props.name, "vram_gb": round(props.total_memory / 1e9, 1)}
    except Exception:
        pass
    return {"cuda": False, "name": "", "vram_gb": 0}


@app.get("/api/models")
async def list_models(arch: str = "sdxl", root: str = ""):
    root = clean_path(root) or _detect_model_root()
    backend = get_family(arch)["backend"]  # one family -> its backend -> its folders
    models = _scan_models(root, backend) if root and os.path.isdir(root) else []
    return {"root": root, "models": models}


@app.get("/api/dataset/list")
async def dataset_list(dir: str, output_dir: str = ""):
    dir = clean_path(dir)
    output_dir = clean_path(output_dir)
    images = []
    if os.path.isdir(dir):
        for p in sorted(glob.glob(os.path.join(dir, "*"))):
            if p.lower().endswith(_IMG_EXTS):
                txt = caption_path(p, output_dir)
                caption = ""
                if os.path.exists(txt):
                    with open(txt, encoding="utf-8") as f:
                        caption = f.read()
                images.append({"path": p, "name": os.path.basename(p), "caption": caption})
    return {"images": images, "count": len(images)}


@app.post("/api/dataset/upload")
async def dataset_upload(dir: str = Form(...), files: list[UploadFile] = File(...)):
    """Save browser-uploaded images into a server-side folder — lets users skip scp:
    pick a target path, drop images, train. Only image files are kept."""
    d = clean_path(dir)
    if not d:
        return {"ok": False, "error": "no target folder", "saved": 0}
    os.makedirs(d, exist_ok=True)
    saved = 0
    for f in files:
        name = os.path.basename(f.filename or "")
        if not name.lower().endswith(_IMG_EXTS):
            continue
        with open(os.path.join(d, name), "wb") as out:
            out.write(await f.read())
        saved += 1
    return {"ok": True, "saved": saved, "dir": d}


_CACHE_HDR = {"Cache-Control": "public, max-age=86400"}


@app.get("/api/dataset/image")
async def dataset_image(path: str):
    if not os.path.isfile(path) or not path.lower().endswith(_IMG_EXTS):
        return Response(status_code=404)
    return FileResponse(path, headers=_CACHE_HDR)


_THUMB_CACHE: dict = {}  # (path, mtime, size) -> jpeg bytes


@app.get("/api/dataset/thumb")
async def dataset_thumb(path: str, size: int = 384):
    """Small thumbnail (JPEG) for the grid: ~16× fewer decoded pixels than
    the full-res -> less browser memory pressure, less flicker. Cached."""
    path = clean_path(path)
    if not os.path.isfile(path) or not path.lower().endswith(_IMG_EXTS):
        return Response(status_code=404)
    try:
        import io

        from PIL import Image

        key = (path, os.path.getmtime(path), size)
        buf = _THUMB_CACHE.get(key)
        if buf is None:
            im = Image.open(path).convert("RGB")
            im.thumbnail((size, size), Image.LANCZOS)
            b = io.BytesIO()
            im.save(b, format="JPEG", quality=82)
            buf = b.getvalue()
            if len(_THUMB_CACHE) > 500:
                _THUMB_CACHE.clear()
            _THUMB_CACHE[key] = buf
        return Response(content=buf, media_type="image/jpeg", headers=_CACHE_HDR)
    except Exception:
        return FileResponse(path, headers=_CACHE_HDR)


@app.post("/api/caption/save")
async def caption_save(body: CaptionSave):
    txt = caption_path(body.path, body.output_dir)
    with open(txt, "w", encoding="utf-8") as f:
        f.write(body.text)
    return {"ok": True, "path": txt}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if _TOKEN and ws.query_params.get("token") != _TOKEN:
        await ws.close(code=1008)  # policy violation
        return
    await ws.accept()
    hub.clients.add(ws)
    try:
        # DO NOT replay the whole history: on a reconnect (frequent when
        # the loop is starved by a GPU job), replaying hundreds of events
        # causes a burst of client-side re-renders (flicker). The client keeps
        # its React state on reconnect -> we only resend the last STATUS.
        last_status = next((e for e in reversed(hub.history) if e.get("type") == "status"), None)
        if last_status is not None:
            await ws.send_text(json.dumps(last_status))
        while True:
            await ws.receive_text()  # keepalive / pings ignored
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(ws)


# ------------------------------------------------------------------ UI web
# Serves the built React frontend for cloud use (Vast): open http://<ip>:8765/
# in a browser = the full UI, no local app. Mounted AFTER the /api and /ws routes
# (they take priority). Absent in dev => nothing is mounted (the UI runs on Vite :1420).
_HERE = os.path.dirname(os.path.abspath(__file__))
_WEB_DIR = os.environ.get("SOMA_WEB_DIR") or next(
    (d for d in (os.path.join(_HERE, "web"), os.path.join(_HERE, "..", "dist"))
     if os.path.isdir(d)), None)
if _WEB_DIR:
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    # ws_ping disabled: during a GPU job the asyncio loop is starved (GIL of the
    # model thread) -> the WebSocket keepalive would skip and cause disconnect/
    # reconnecter en boucle. Sans ping serveur, la connexion tient (local, TCP).
    uvicorn.run(
        app, host="127.0.0.1", port=8765, log_level="warning",
        ws_ping_interval=None, ws_ping_timeout=None,
    )
