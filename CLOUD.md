# SOMA on the cloud (Vast.ai) — Docker + CLI

The SOMA engine runs inside a GPU Docker image. Two ways to use it:
- **Headless CLI**: run a training in a single command.
- **Web UI**: open the full SOMA interface in a browser, served by the cloud machine.

## 1. Build the image

```bash
docker build -t soma:latest .          # builds the UI (Vite) + engine (torch cu128)
# non-Blackwell GPU / older driver:
docker build --build-arg CUDA_CHANNEL=cu124 -t soma:latest .
```

(Or push to a registry / Docker Hub so Vast.ai can pull it directly.)

## 2. On Vast.ai

Pick a GPU machine (VRAM depending on the architecture — see `soma archs`), image = `soma:latest`.
Mount your data/weights as volumes, and expose the port if you want the UI.

### A. Training via the CLI (simplest)

```bash
docker run --gpus all \
  -v /data:/data -v /models:/models -v /cache:/cache/hf \
  soma:latest train \
    --arch flux --base /models/flux1-dev.safetensors \
    --dataset /data/mychar --output /data/out \
    --precision nf4 --rank 16 --steps 1500 --resolution 768 \
    --project mychar
```

- `--arch`: see `docker run soma archs` (25 families).
- `--base`: local path (mounted weights) **or** a HF repo (`Qwen/Qwen-Image`, `stabilityai/stable-diffusion-3.5-medium`…) — downloaded into `/cache/hf`.
- `--config run.json`: pass everything from a JSON file (same keys as `TrainConfig`).
- Styled terminal output: live progress bar + loss + ETA.

### B. Web UI (GUI in the browser)

```bash
docker run --gpus all -p 8765:8765 \
  -v /data:/data -v /models:/models -v /cache:/cache/hf \
  soma:latest serve
```

> **Auto-serve (Vast):** with no command and no TTY (e.g. Vast's *Docker ENTRYPOINT* launch
> mode), the image **starts the web server automatically** on `:8765` — no need to pass `serve`.
> A real terminal (`docker run -it soma`) still opens the interactive TUI launcher. Force serve
> anywhere with `-e SOMA_SERVE=1` (host/port via `SOMA_HOST` / `SOMA_PORT`).

Two ways to reach the UI from your PC:

**Direct (simple, public port)** — expose port 8765 on Vast, then open the address it gives you
(`http://<vast-ip>:<mapped-port>/`). The UI talks to the engine on the **same origin** (no config,
even if Vast maps it to a different port number).

**SSH tunnel (private, nothing exposed)** — the image bundles an SSH server that **only starts if a
public key is provided** (Vast passes `$PUBLIC_KEY` automatically when you add your key to the
instance). From your PC:
```bash
ssh -L 8765:localhost:8765 root@<vast-ip> -p <vast-ssh-port>
# then open http://localhost:8765/  (encrypted traffic, port never public)
```
> SSH is **opt-in**: without a key, no `sshd` runs (same behavior as before).

### C. Local desktop app → cloud engine (advanced)

Keep the SOMA app local and point it at the Vast engine: in the browser/app console,
`localStorage.setItem("soma.engineUrl", "http://<vast-ip>:8765")`.

## Notes

- **Large models (20B+)**: nf4 quantization loads the weights as bf16 in RAM while quantizing
  (~2× the fp16 size). Pick a machine with enough **CPU RAM** (not just VRAM), or it OOMs on load.
- Mounted HF cache (`/cache/hf`) → avoids re-downloading the large text encoders between runs.
- `soma serve` also exposes the API (`POST /api/train/start`, WS `/ws`) for automation.
