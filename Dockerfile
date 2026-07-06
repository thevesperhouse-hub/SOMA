# SOMA — GPU training image (Vast.ai / cloud). Python engine + CLI + web UI.
# Build : docker build -t soma:latest .
# Train : docker run --gpus all -v /data:/data -v /models:/models soma \
#           train --arch flux --base /models/flux1-dev.safetensors --dataset /data/mychar --steps 1200 --precision nf4
# Web UI: docker run --gpus all -p 8765:8765 -v /data:/data -v /models:/models soma serve
#         then open http://<ip>:8765/ in a browser (full UI, no local app)
#
# CUDA 12.8 (cu128) = works on Blackwell (RTX 50xx) + recent Ampere/Ada/Hopper.
# Override: --build-arg CUDA_CHANNEL=cu124

# ---------- Stage 1: build the web UI (React/Vite) ----------
FROM node:20-slim AS webbuild
WORKDIR /web
COPY package.json package-lock.json* ./
RUN npm ci 2>/dev/null || npm install
COPY . .
RUN npm run build   # produces /web/dist

# ---------- Stage 2: GPU engine ----------
# Ubuntu 24.04 ships python3.12 natively (no deadsnakes PPA = more robust build).
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ARG CUDA_CHANNEL=cu128
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/cache/hf \
    SOMA_MODEL_ROOT=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git ca-certificates curl && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps in separate layers, ordered least -> most frequently changed, so editing the
# small server deps never re-downloads torch. NB: do NOT upgrade pip (the apt one has no
# RECORD file -> "Cannot uninstall pip").
# 1) torch (biggest, changes only with CUDA_CHANNEL)
RUN python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/${CUDA_CHANNEL} && \
    python -m pip install hf_transfer
# 2) heavy training deps (diffusers / transformers / peft / bitsandbytes …)
COPY engine/requirements-train.txt ./
RUN python -m pip install -r requirements-train.txt
# 3) light server deps (fastapi / uvicorn / multipart / rich) — cheap to rebuild
COPY engine/requirements-base.txt ./
RUN python -m pip install -r requirements-base.txt

# SSH server (optional, for a private tunnel on Vast). Separate layer AFTER pip so the
# large torch layer stays cached (fast rebuilds).
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server && \
    rm -rf /var/lib/apt/lists/* && mkdir -p /run/sshd && \
    sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/; s/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

# engine code + bundled configs, + built web UI (stage 1) + entrypoint
COPY engine/ /app/
COPY --from=webbuild /web/dist /app/web
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN sed -i 's/\r$//' /docker-entrypoint.sh && chmod +x /docker-entrypoint.sh

# 8765 = web UI/API ; 22 = SSH (only starts if a public key is provided)
EXPOSE 8765 22
ENTRYPOINT ["/docker-entrypoint.sh"]
