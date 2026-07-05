# SOMA — image d'entraînement GPU (Vast.ai / cloud). Moteur Python + CLI + UI web.
# Build : docker build -t soma:latest .
# Train : docker run --gpus all -v /data:/data -v /models:/models soma \
#           train --arch flux --base /models/flux1-dev.safetensors --dataset /data/mychar --steps 1200 --precision nf4
# UI web: docker run --gpus all -p 8765:8765 -v /data:/data -v /models:/models soma serve
#         puis ouvrir http://<ip>:8765/ dans un navigateur (toute l'UI, sans app locale)
#
# CUDA 12.8 (cu128) = compatible Blackwell (RTX 50xx) + Ampere/Ada/Hopper récents.
# Override : --build-arg CUDA_CHANNEL=cu124

# ---------- Stage 1 : build de l'UI web (React/Vite) ----------
FROM node:20-slim AS webbuild
WORKDIR /web
COPY package.json package-lock.json* ./
RUN npm ci 2>/dev/null || npm install
COPY . .
RUN npm run build   # produit /web/dist

# ---------- Stage 2 : moteur GPU ----------
# Ubuntu 24.04 embarque python3.12 nativement (pas de PPA deadsnakes = build + robuste).
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

# deps (couche cachable) : torch cu128 d'abord, puis le reste.
# NB : on n'upgrade PAS pip (celui d'apt n'a pas de RECORD -> "Cannot uninstall pip").
COPY engine/requirements-base.txt engine/requirements-train.txt ./
RUN python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/${CUDA_CHANNEL} && \
    python -m pip install hf_transfer && \
    python -m pip install -r requirements-base.txt -r requirements-train.txt

# code moteur + configs embarquées, + UI web buildée (stage 1)
COPY engine/ /app/
COPY --from=webbuild /web/dist /app/web

EXPOSE 8765
ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]
