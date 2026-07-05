# SOMA sur le cloud (Vast.ai) — Docker + CLI

Le moteur SOMA tourne dans une image Docker GPU. Deux usages :
- **CLI headless** : lancer un entraînement en une commande.
- **UI web** : ouvrir toute l'interface SOMA dans un navigateur, servie par la machine cloud.

## 1. Build de l'image

```bash
docker build -t soma:latest .          # build UI (Vite) + moteur (torch cu128)
# GPU non-Blackwell / driver plus ancien :
docker build --build-arg CUDA_CHANNEL=cu124 -t soma:latest .
```

(Ou push sur un registry / Docker Hub pour que Vast.ai la tire directement.)

## 2. Sur Vast.ai

Choisir une machine GPU (VRAM selon l'archi — voir `soma archs`), image = `soma:latest`.
Monter tes données/poids en volumes et exposer le port si tu veux l'UI.

### A. Entraînement en CLI (le plus simple)

```bash
docker run --gpus all \
  -v /data:/data -v /models:/models -v /cache:/cache/hf \
  soma:latest train \
    --arch flux --base /models/flux1-dev.safetensors \
    --dataset /data/mychar --output /data/out \
    --precision nf4 --rank 16 --steps 1500 --resolution 768 \
    --project mychar
```

- `--arch` : voir `docker run soma archs` (25 familles).
- `--base` : chemin local (poids montés) **ou** repo HF (`Qwen/Qwen-Image`, `stabilityai/stable-diffusion-3.5-medium`…) — téléchargé dans `/cache/hf`.
- `--config run.json` : tout passer par un fichier JSON (mêmes clés que `TrainConfig`).
- Rendu terminal stylisé : barre de progression + loss + ETA en direct.

### B. UI web (GUI dans le navigateur)

```bash
docker run --gpus all -p 8765:8765 \
  -v /data:/data -v /models:/models -v /cache:/cache/hf \
  soma:latest serve
```

Puis ouvrir **`http://<ip-vast>:8765/`** → toute l'UI SOMA (thèmes, courbe de loss live,
Dataset, XP…). L'UI parle au moteur en **même origine** (aucune config).

> Astuce sécurité : le port est public. Pour restreindre, passer par un tunnel SSH
> (`ssh -L 8765:localhost:8765 user@vast`) et ouvrir `http://localhost:8765/`.

### C. App desktop locale → moteur cloud (option avancée)

Garder l'app SOMA en local et la pointer sur le moteur Vast : dans la console du
navigateur/app, `localStorage.setItem("soma.engineUrl", "http://<ip-vast>:8765")`.

## Notes

- **Modèles lourds (20B+)** : la quantization nf4 charge les poids en bf16 en RAM le
  temps de quantifier (~2× la taille fp16). Prendre une machine avec assez de **RAM CPU**
  (pas que de la VRAM), sinon OOM au chargement.
- Cache HF monté (`/cache/hf`) → ne re-télécharge pas les gros text encoders entre runs.
- `soma serve` sert aussi l'API (`POST /api/train/start`, WS `/ws`) pour l'automatisation.
