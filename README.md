<div align="center">

```
██████   ██████   ███    ███   █████
██       ██  ██   ████  ████  ██   ██
██████   ██  ██   ██ ██ ██   ███████
    ██   ██  ██   ██    ██   ██   ██
██████   ██████   ██    ██   ██   ██
```

### LoRA Training Studio

**Train character-identity LoRAs across 25 image architectures — one homemade engine, from your terminal to the cloud.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-akiraxan%2Fsoma-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/r/akiraxan/soma)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

**SOMA** is a self-hosted LoRA training studio. It uses its **own** `diffusers` + `peft` training loops
(no kohya, no ai-toolkit wrapper) so every architecture is a small, readable, hackable file. Drive it from a
**styled terminal launcher**, a **headless CLI**, or a **web UI served straight from the engine** — locally on
your GPU or in the cloud (Vast.ai, RunPod…).

> Built and tested on an RTX 5080 (Blackwell / CUDA 12.8). Verified end-to-end from `docker run` to a
> `.safetensors` LoRA on disk.

<!-- 👉 tip: drop a GIF of the TUI cockpit and a web-UI screenshot here to make the repo pop -->

## ✨ Highlights

- **25 image architectures** — SDXL, Flux, Qwen-Image, SD 3.5, Chroma, Lumina2, Sana, PixArt, HunyuanImage… (full list below)
- **Our own engine** — flow-matching *and* epsilon objectives, per-model pluggable trainers, no black-box wrapper
- **Fits 16 GB** — nf4 / int8 QLoRA, gradient checkpointing, cached latents + text embeddings
- **3 ways to drive it** — interactive **TUI launcher**, **CLI** (`soma train …`), or **web UI** (`soma serve`)
- **Cloud-ready** — one Docker image, GPU-verified, published on Docker Hub
- **ComfyUI / kohya export** — LoRAs come out ready to use
- **Contributor-friendly by design** — adding a new architecture is essentially *one file* ([see below](#-add-an-architecture-good-first-pr))

## 🖥️ The terminal, not an afterthought

The launcher opens on an animated SOMA logo, then a live **training cockpit** — a braille loss curve, the sample
**preview rendered as an image in the terminal**, and live GPU telemetry:

```
┌───────────── flux · my-character ──────────────────────────────────┐
│  ⠿ ████████████████████░░░░░░  72%  144/200   0:03  0:01           │
│  ┌──── loss 0.041 ─────────────┐ ┌──── preview ─────────────────┐  │
│  │ 0.160 ⣀                     │ │ ▀▀▀▀▀▀  (the character face   │  │
│  │       ⠉⠢⣀                   │ │ ▀▀▀▀▀▀   emerging, step/step) │  │
│  │ 0.038    ⠉⠒⠤⣀⣀___  ↓        │ │ ▀▀▀▀▀▀                        │  │
│  └─────────────────────────────┘ └──────────────────────────────┘  │
│  ┌─ gpu ──────────┐┌────────────── log ───────────────────────┐    │
│  │ load ████░ 8%  ││  · Optimizer: AdamW8bit                   │    │
│  │ vram █░░░ 1.4/16││  · generating a preview…                 │    │
│  └────────────────┘└───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick start

### Docker (recommended)

```bash
# web UI — full interface in your browser
docker run --gpus all -p 8765:8765 \
  -v $PWD/data:/data -v $PWD/models:/models -v $PWD/cache:/cache/hf \
  akiraxan/soma serve
# → open http://localhost:8765/

# or headless training
docker run --gpus all -v $PWD/data:/data -v $PWD/cache:/cache/hf \
  akiraxan/soma train --arch flux --base black-forest-labs/FLUX.1-dev \
  --dataset /data/mychar --precision nf4 --rank 16 --steps 1500
```

### From source

```bash
git clone https://github.com/thevesperhouse-hub/SOMA.git && cd SOMA
python engine/bootstrap.py            # creates the venv + installs torch (cu128) & deps
npm install && npm run build          # builds the web UI (optional)

# interactive TUI launcher
engine/.venv/bin/python engine/cli.py         #  (Windows: engine\.venv\Scripts\python.exe)

# or a direct run
engine/.venv/bin/python engine/cli.py train --arch sdxl --base /path/model.safetensors --dataset ./mychar --steps 1200
```

`--base` accepts a **Hugging Face repo id** or a **local checkpoint**. `--config run.json` loads everything from a file.

## 🧩 Supported architectures

| Family | Backend | Objective | Text encoder(s) |
|---|---|---|---|
| SDXL · Pony · Illustrious · NoobAI (eps & v-pred) | `sdxl` | epsilon / v-pred | CLIP-L + CLIP-G |
| SD 1.5 | `sd15` | epsilon | CLIP-L |
| Z-Image (Turbo / full / De-Turbo) | `zimage` | flow | Qwen3 |
| Flux.1-dev · FLUX.1 Krea | `flux` | flow | CLIP-L + T5 |
| Qwen-Image (+ 2512) | `qwen` | flow | Qwen2.5-VL |
| Chroma | `chroma` | flow | T5 |
| Lumina2 | `lumina2` | flow | Gemma-2 |
| SD 3.5 | `sd3` | flow | CLIP-L + CLIP-G + T5 |
| Sana | `sana` | flow | Gemma-2 |
| PixArt-Sigma | `pixart` | epsilon | T5 |
| Bria 3.x | `bria` | flow | T5 |
| AuraFlow | `auraflow` | flow | UMT5 |
| CogView4 | `cogview4` | flow | GLM-4 |
| Ovis-Image | `ovis` | flow | Qwen3 |
| Kolors | `kolors` | epsilon | ChatGLM3 |
| HunyuanImage | `hunyuanimage` | flow | Qwen2.5-VL |
| PRX | `prx` | flow | T5Gemma |

Run `soma archs` for the live list. *(Video — Wan2.2 — and more image models are on the roadmap.)*

## 🏗️ How it works

```
engine/
├── families.py        # single source of truth: every model = one registry entry
├── trainer.py         # dispatch: family → backend → pluggable trainer
├── <arch>_trainer.py  # one file per backend (flux, qwen, chroma, sana, …)
├── quant.py           # shared nf4/int8 QLoRA helpers
├── captioner.py       # JoyCaption dataset captioning
├── server.py          # FastAPI + WebSocket API, serves the web UI
├── cli.py / launcher.py   # headless CLI + styled TUI
└── model_configs/     # bundled DiT configs (offline loading)
src/                   # React + Vite + Tailwind web UI (themes, live loss chart, XP)
```

Each trainer wires the model's VAE, text encoder(s), noising objective and LoRA targets via `diffusers`/`peft`.
The registry + dispatch means the UI, CLI and model scanner all derive from one place.

## 🤝 Add an architecture (good first PR)

SOMA is built so a new model is roughly **one file**:

1. Read the model's `diffusers` pipeline source (text embeds, transformer forward, VAE normalization, target sign).
2. Copy the closest `*_trainer.py` and adapt those four things.
3. Add one entry to `FAMILIES` in `families.py`, and one `elif` in `trainer.py`.
4. Validate the LoRA target names with `init_empty_weights()` + `named_modules()` (no full run needed).

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the step-by-step guide. Issues tagged `good first issue` are new architectures waiting for a home.

## 🗺️ Roadmap

- [ ] Video architectures (Wan2.2 T2V/I2V)
- [ ] More image models (FLUX.2, OmniGen2, Krea-2, HiDream, ERNIE, Nucleus…)
- [ ] Per-architecture slim Docker images
- [ ] Desktop installer (Tauri) bundling the engine
- [ ] Identity-similarity scoring ("Identity Lock")

## 📄 License

[MIT](LICENSE) — do what you want, no warranty. Contributions welcome under the same license.

<div align="center"><sub>Made with a lot of GPU heat. ❤️</sub></div>
