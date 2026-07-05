"""Registre des familles de modèles — SOURCE UNIQUE de vérité.

Une "famille" = un type de modèle sélectionnable dans l'UI. Elle déclare :
  - backend    : quel moteur d'entraînement l'exécute ("sdxl" | "zimage")
  - prediction : objectif de la loss ("epsilon" | "v_prediction" | "flow")
  - zsnr       : zero-terminal-SNR (modèles v-pred type NoobAI v-pred)
  - resolution : résolution par défaut
  - default_base : base_model par défaut (repo HF) ; "" = l'utilisateur choisit un checkpoint
  - prompt_hint  : convention de prompt de la famille (affichée dans l'UI)

Pony / Illustrious / NoobAI sont des dérivés SDXL -> même backend "sdxl". Seul
NoobAI v-pred change l'objectif (v_prediction + zsnr). Ajouter une famille =
1 entrée ici ; le dispatch (trainer) et l'UI (via /api/families) en découlent.
"""

FAMILIES = [
    {
        "id": "sdxl", "label": "SDXL", "backend": "sdxl", "prediction": "epsilon",
        "resolution": 1024, "default_base": "stabilityai/stable-diffusion-xl-base-1.0",
        "prompt_hint": "a portrait photo of <token> person, natural light",
        "params_b": 2.6, "quantizable": False,
    },
    {
        "id": "pony", "label": "Pony", "backend": "sdxl", "prediction": "epsilon",
        "resolution": 1024, "default_base": "",
        "prompt_hint": "score_9, score_8_up, score_7_up, <token>, ...",
        "params_b": 2.6, "quantizable": False,
    },
    {
        "id": "illustrious", "label": "Illustrious", "backend": "sdxl", "prediction": "epsilon",
        "resolution": 1024, "default_base": "",
        "prompt_hint": "masterpiece, best quality, <token>, 1girl, ...",
        "params_b": 2.6, "quantizable": False,
    },
    {
        "id": "noobai", "label": "NoobAI (eps)", "backend": "sdxl", "prediction": "epsilon",
        "resolution": 1024, "default_base": "",
        "prompt_hint": "masterpiece, best quality, <token>, 1girl, ...",
        "params_b": 2.6, "quantizable": False,
    },
    {
        "id": "noobai_vpred", "label": "NoobAI (v-pred)", "backend": "sdxl",
        "prediction": "v_prediction", "zsnr": True,
        "resolution": 1024, "default_base": "",
        "prompt_hint": "masterpiece, best quality, <token>, 1girl, ...",
        "params_b": 2.6, "quantizable": False,
    },
    {
        "id": "zimage", "label": "Z-Image Turbo", "backend": "zimage", "prediction": "flow",
        "resolution": 768, "default_base": "Tongyi-MAI/Z-Image-Turbo",
        "prompt_hint": "a portrait photo of <token> person",
        "params_b": 6.0, "quantizable": True,
    },
    {
        "id": "flux", "label": "Flux.1-dev", "backend": "flux", "prediction": "flow",
        "resolution": 512, "default_base": "",
        "prompt_hint": "a photo of <token> person, cinematic lighting",
        "params_b": 12.0, "quantizable": True,
    },
    {
        # FLUX.1 Krea [dev] = finetune BFL/Krea de l'ARCHI Flux.1-dev (esthétique
        # photo, moins "IA"). Même DiT -> backend flux inchangé, juste un preset.
        "id": "flux_krea", "label": "FLUX.1 Krea [dev]", "backend": "flux", "prediction": "flow",
        "resolution": 512, "default_base": "",
        "prompt_hint": "a photo of <token>, natural aesthetic, soft light, film grain",
        "params_b": 12.0, "quantizable": True,
    },
    {
        # Qwen-Image (MMDiT ~20B, texte encodé par Qwen2.5-VL). Backend dédié :
        # VAE 3D + embeddings Qwen2.5-VL, flow-matching, guidance=None.
        "id": "qwen_image", "label": "Qwen-Image", "backend": "qwen", "prediction": "flow",
        "resolution": 1024, "default_base": "",
        "prompt_hint": "a photo of <token> person, natural light",
        "params_b": 20.0, "quantizable": True,
    },
    {
        # Qwen-Image 2512 = checkpoint plus récent, MÊME QwenImageTransformer2DModel
        # -> backend qwen inchangé, juste un preset.
        "id": "qwen_image_2512", "label": "Qwen-Image 2512", "backend": "qwen", "prediction": "flow",
        "resolution": 1024, "default_base": "",
        "prompt_hint": "a photo of <token> person, natural light",
        "params_b": 20.0, "quantizable": True,
    },
    {
        # Z-Image (full, NON distillé) — même archi que Turbo, backend zimage inchangé.
        # Non-turbo = pas de souci de de-distillation à l'entraînement de LoRA.
        "id": "zimage_full", "label": "Z-Image", "backend": "zimage", "prediction": "flow",
        "resolution": 768, "default_base": "Tongyi-MAI/Z-Image",
        "prompt_hint": "a portrait photo of <token> person",
        "params_b": 6.0, "quantizable": True,
    },
    {
        # Z-Image De-Turbo (ostris) — Turbo dé-distillé, même archi, backend zimage.
        "id": "zimage_deturbo", "label": "Z-Image De-Turbo", "backend": "zimage", "prediction": "flow",
        "resolution": 768, "default_base": "ostris/Z-Image-De-Turbo",
        "prompt_hint": "a portrait photo of <token> person",
        "params_b": 6.0, "quantizable": True,
    },
    {
        # Chroma = Flux élagué + dé-distillé, T5-only (pas de CLIP), pas de guidance.
        # Réutilise le Flux AE + T5 locaux. Backend dédié.
        "id": "chroma", "label": "Chroma", "backend": "chroma", "prediction": "flow",
        "resolution": 512, "default_base": "",
        "prompt_hint": "a photo of <token> person, cinematic lighting",
        "params_b": 8.9, "quantizable": True,
    },
    {
        # SD 1.5 — UNet epsilon, un seul CLIP (768), 512px. Backend dédié (≠ SDXL :
        # pas de double text encoder ni d'add_time_ids). Petit, pas de quantization.
        "id": "sd15", "label": "SD 1.5", "backend": "sd15", "prediction": "epsilon",
        "resolution": 512, "default_base": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "prompt_hint": "a photo of <token> person",
        "params_b": 0.86, "quantizable": False,
    },
    {
        # Lumina2 (Alpha-VLLM/Lumina-Image-2.0) — DiT flow, texte Gemma-2, VAE Flux AE.
        # Convention flow = Z-Image (timestep 1-sigma, cible x1-x0). Backend dédié.
        "id": "lumina2", "label": "Lumina2", "backend": "lumina2", "prediction": "flow",
        "resolution": 1024, "default_base": "",
        "prompt_hint": "a portrait photo of <token> person, detailed",
        "params_b": 2.6, "quantizable": True,
    },
    {
        # PRX (Photoroom/prxpixel-t2i) — DiT flow compact, texte T5Gemma, VAE KL 16ch.
        # Distribué en repo diffusers (pas single-file) -> default_base = le repo.
        "id": "prx", "label": "PRX (PRXPixel)", "backend": "prx", "prediction": "flow",
        "resolution": 512, "default_base": "Photoroom/prxpixel-t2i",
        "prompt_hint": "a photo of <token> person",
        "params_b": 1.0, "quantizable": True,
    },
    {
        # SD 3.5 — MMDiT flow, 3 text encoders (CLIP-L + CLIP-G + T5). Repo diffusers.
        "id": "sd3", "label": "SD 3.5", "backend": "sd3", "prediction": "flow",
        "resolution": 1024, "default_base": "stabilityai/stable-diffusion-3.5-medium",
        "prompt_hint": "a photo of <token> person",
        "params_b": 2.5, "quantizable": True,
    },
    {
        # Sana — DiT flow linéaire, Gemma-2, VAE DC-AE 32×. Repo diffusers, très rapide.
        "id": "sana", "label": "Sana", "backend": "sana", "prediction": "flow",
        "resolution": 1024, "default_base": "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        "prompt_hint": "a photo of <token> person",
        "params_b": 1.6, "quantizable": True,
    },
    {
        # PixArt-Sigma — DiT epsilon, T5-only, VAE KL 4ch. Repo diffusers, léger.
        "id": "pixart", "label": "PixArt-Sigma", "backend": "pixart", "prediction": "epsilon",
        "resolution": 1024, "default_base": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "prompt_hint": "a photo of <token> person",
        "params_b": 0.6, "quantizable": True,
    },
    {
        # Bria 3.x — archi Flux T5-only (no pooled/guidance), VAE KL 16ch. Repo diffusers.
        "id": "bria", "label": "Bria 3.2", "backend": "bria", "prediction": "flow",
        "resolution": 1024, "default_base": "briaai/BRIA-3.2",
        "prompt_hint": "a photo of <token> person",
        "params_b": 4.0, "quantizable": True,
    },
    {
        # AuraFlow — MMDiT flow, texte UMT5, VAE KL 4ch. Repo diffusers.
        "id": "auraflow", "label": "AuraFlow", "backend": "auraflow", "prediction": "flow",
        "resolution": 1024, "default_base": "fal/AuraFlow-v0.3",
        "prompt_hint": "a photo of <token> person",
        "params_b": 6.8, "quantizable": True,
    },
    {
        # CogView4 — DiT flow, texte GLM-4-9B, VAE KL, micro-cond SDXL. Repo diffusers.
        "id": "cogview4", "label": "CogView4", "backend": "cogview4", "prediction": "flow",
        "resolution": 1024, "default_base": "THUDM/CogView4-6B",
        "prompt_hint": "a photo of <token> person",
        "params_b": 6.0, "quantizable": True,
    },
    {
        # Ovis-Image — archi Flux, texte Qwen3 (chat template), no guidance. Repo diffusers.
        "id": "ovis", "label": "Ovis-Image", "backend": "ovis", "prediction": "flow",
        "resolution": 1024, "default_base": "",
        "prompt_hint": "a photo of <token> person",
        "params_b": 3.0, "quantizable": True,
    },
    {
        # Kolors — UNet SDXL + ChatGLM3-6B, epsilon. Repo diffusers.
        "id": "kolors", "label": "Kolors", "backend": "kolors", "prediction": "epsilon",
        "resolution": 1024, "default_base": "Kwai-Kolors/Kolors-diffusers",
        "prompt_hint": "a photo of <token> person",
        "params_b": 2.6, "quantizable": False,
    },
    {
        # HunyuanImage — DiT flow ~17B, Qwen2.5-VL (+ByT5 ignoré), VAE 32×. Repo diffusers.
        "id": "hunyuanimage", "label": "HunyuanImage", "backend": "hunyuanimage", "prediction": "flow",
        "resolution": 1024, "default_base": "tencent/HunyuanImage-2.1",
        "prompt_hint": "a photo of <token> person",
        "params_b": 17.0, "quantizable": True,
    },
]

_BY_ID = {f["id"]: f for f in FAMILIES}


def get_family(arch: str) -> dict:
    """Renvoie la famille (fallback SDXL si inconnue)."""
    return _BY_ID.get((arch or "sdxl"), _BY_ID["sdxl"])


def soma_meta(cfg, family, step) -> dict:
    """Métadonnées embarquées dans le .safetensors du LoRA (relues par /api/checkpoints
    pour lister par archi/nom/date). Valeurs = chaînes (contrainte safetensors)."""
    import datetime
    import os

    fam = family or {}
    base = os.path.basename((getattr(cfg, "base_model", "") or "").rstrip("/\\"))
    return {
        "soma_arch": fam.get("id", getattr(cfg, "arch", "")) or "",
        "soma_label": fam.get("label", "") or "",
        "soma_base": base,
        "soma_steps": str(step),
        "soma_date": datetime.datetime.now().isoformat(timespec="seconds"),
    }
