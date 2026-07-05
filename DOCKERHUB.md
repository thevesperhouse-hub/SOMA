# SOMA — LoRA Training Studio

**Train character-identity LoRAs across 25 image architectures**, powered by a homemade
`diffusers` + `peft` engine (no kohya, no ai-toolkit). Use it as a **headless CLI**, a
**web UI** (served straight from the container), or a **styled terminal launcher**.

Built to run from laptop to cloud GPU (Vast.ai, RunPod…). CUDA 12.8 (`cu128`) →
works on **RTX 50xx (Blackwell)** + Ampere / Ada / Hopper.

---

## 🚀 Quick start

### Web UI (full interface in your browser)
```bash
docker run --gpus all -p 8765:8765 \
  -v /workspace/data:/data -v /workspace/models:/models -v /workspace/cache:/cache/hf \
  akiraxan/soma serve
# → open http://<ip>:8765/
```

### Train from the CLI
```bash
docker run --gpus all -v /workspace/data:/data -v /workspace/cache:/cache/hf \
  akiraxan/soma train \
    --arch flux --base black-forest-labs/FLUX.1-dev \
    --dataset /data/mychar --precision nf4 --rank 16 --steps 1500
```

`--base` accepts a **Hugging Face repo** (downloaded to `/cache/hf`) or a **local file mounted** under `/models`.

### List the architectures
```bash
docker run --rm akiraxan/soma archs
```

---

## 🧩 Supported architectures

**Image (25)** — SDXL · Pony · Illustrious · NoobAI (eps + v-pred) · SD 1.5 · Z-Image (Turbo / full / De-Turbo)
· Flux.1-dev · FLUX.1 Krea · Qwen-Image (+2512) · Chroma · Lumina2 · SD 3.5 · Sana · PixArt-Sigma
· Bria · AuraFlow · CogView4 · Ovis-Image · Kolors · HunyuanImage · PRX

Flow-matching **and** epsilon backends, **nf4 / int8** weight quantization (QLoRA) to fit 16 GB,
latent + text-embedding caching, and **ComfyUI / kohya** LoRA export out of the box.

---

## 📦 Commands

| Command | Purpose |
|---|---|
| `serve` | API + **web UI** on `:8765` |
| `train` | run a training (flags or `--config run.json`) |
| `caption` | caption a dataset (JoyCaption) |
| `archs` | list architectures |
| *(none)* | interactive **TUI launcher** (logo, menu, live cockpit) |

## 🔧 Volumes & env

| Volume | Purpose |
|---|---|
| `/data` | datasets (images + `.txt` captions) and outputs |
| `/models` | local checkpoints (optional) |
| `/cache/hf` | Hugging Face cache (avoids re-downloading text encoders) |

`SOMA_MODEL_ROOT=/models` · `HF_HOME=/cache/hf`

---

## ⚠️ Notes

- **`--gpus all` required** (NVIDIA runtime). Verified end-to-end: model load → GPU training → LoRA export.
- **Large models (20B+)**: nf4 quantization loads the weights as bf16 in RAM while quantizing → make sure you have enough **CPU RAM** (≥ 64 GB), not just VRAM.
- Public port in `serve` mode → use an **SSH tunnel** for security (`ssh -L 8765:localhost:8765 user@host`).
