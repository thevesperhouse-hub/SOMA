"""Model family registry — the SINGLE SOURCE of truth.

A "family" = a model type selectable in the UI. It declares:
  - backend    : which training engine runs it ("sdxl" | "zimage" | …)
  - prediction : the loss objective ("epsilon" | "v_prediction" | "flow")
  - zsnr       : zero-terminal-SNR (v-pred models like NoobAI v-pred)
  - resolution : default resolution
  - default_base : default base_model (HF repo); "" = the user picks a checkpoint
  - prompt_hint  : the family's prompt convention (shown in the UI)

Pony / Illustrious / NoobAI are SDXL derivatives -> same "sdxl" backend. Only
NoobAI v-pred changes the objective (v_prediction + zsnr). Adding a family =
one entry here; the dispatch (trainer) and the UI (via /api/families) follow.
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
        # FLUX.1 Krea [dev] = BFL/Krea finetune of the Flux.1-dev ARCH (photo
        # aesthetic, less "AI"-looking). Same DiT -> flux backend unchanged, just a preset.
        "id": "flux_krea", "label": "FLUX.1 Krea [dev]", "backend": "flux", "prediction": "flow",
        "resolution": 512, "default_base": "",
        "prompt_hint": "a photo of <token>, natural aesthetic, soft light, film grain",
        "params_b": 12.0, "quantizable": True,
    },
    {
        # Qwen-Image (MMDiT ~20B, text encoded by Qwen2.5-VL). Dedicated backend:
        # 3D VAE + Qwen2.5-VL embeddings, flow-matching, guidance=None.
        "id": "qwen_image", "label": "Qwen-Image", "backend": "qwen", "prediction": "flow",
        "resolution": 1024, "default_base": "",
        "prompt_hint": "a photo of <token> person, natural light",
        "params_b": 20.0, "quantizable": True,
    },
    {
        # Qwen-Image 2512 = newer checkpoint, SAME QwenImageTransformer2DModel
        # -> qwen backend unchanged, just a preset.
        "id": "qwen_image_2512", "label": "Qwen-Image 2512", "backend": "qwen", "prediction": "flow",
        "resolution": 1024, "default_base": "",
        "prompt_hint": "a photo of <token> person, natural light",
        "params_b": 20.0, "quantizable": True,
    },
    {
        # Z-Image (full, NON distilled) — same arch as Turbo, zimage backend unchanged.
        # Non-turbo = no de-distillation concern when training a LoRA.
        "id": "zimage_full", "label": "Z-Image", "backend": "zimage", "prediction": "flow",
        "resolution": 768, "default_base": "Tongyi-MAI/Z-Image",
        "prompt_hint": "a portrait photo of <token> person",
        "params_b": 6.0, "quantizable": True,
    },
    {
        # Z-Image De-Turbo (ostris) — de-distilled Turbo, same arch, zimage backend.
        "id": "zimage_deturbo", "label": "Z-Image De-Turbo", "backend": "zimage", "prediction": "flow",
        "resolution": 768, "default_base": "ostris/Z-Image-De-Turbo",
        "prompt_hint": "a portrait photo of <token> person",
        "params_b": 6.0, "quantizable": True,
    },
    {
        # Chroma = pruned + de-distilled Flux, T5-only (no CLIP), no guidance.
        # Reuses the local Flux AE + T5. Dedicated backend.
        "id": "chroma", "label": "Chroma", "backend": "chroma", "prediction": "flow",
        "resolution": 512, "default_base": "",
        "prompt_hint": "a photo of <token> person, cinematic lighting",
        "params_b": 8.9, "quantizable": True,
    },
    {
        # SD 1.5 — epsilon UNet, a single CLIP (768), 512px. Dedicated backend (≠ SDXL:
        # no dual text encoder, no add_time_ids). Small, no quantization.
        "id": "sd15", "label": "SD 1.5", "backend": "sd15", "prediction": "epsilon",
        "resolution": 512, "default_base": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "prompt_hint": "a photo of <token> person",
        "params_b": 0.86, "quantizable": False,
    },
    {
        # Lumina2 (Alpha-VLLM/Lumina-Image-2.0) — flow DiT, Gemma-2 text, Flux AE VAE.
        # Flow convention = Z-Image (timestep 1-sigma, target x1-x0). Dedicated backend.
        "id": "lumina2", "label": "Lumina2", "backend": "lumina2", "prediction": "flow",
        "resolution": 1024, "default_base": "",
        "prompt_hint": "a portrait photo of <token> person, detailed",
        "params_b": 2.6, "quantizable": True,
    },
    {
        # PRX (Photoroom/prxpixel-t2i) — compact flow DiT, T5Gemma text, KL VAE 16ch.
        # Distributed as a diffusers repo (not single-file) -> default_base = the repo.
        "id": "prx", "label": "PRX (PRXPixel)", "backend": "prx", "prediction": "flow",
        "resolution": 512, "default_base": "Photoroom/prxpixel-t2i",
        "prompt_hint": "a photo of <token> person",
        "params_b": 1.0, "quantizable": True,
    },
    {
        # SD 3.5 — MMDiT flow, 3 text encoders (CLIP-L + CLIP-G + T5). Diffusers repo.
        "id": "sd3", "label": "SD 3.5", "backend": "sd3", "prediction": "flow",
        "resolution": 1024, "default_base": "stabilityai/stable-diffusion-3.5-medium",
        "prompt_hint": "a photo of <token> person",
        "params_b": 2.5, "quantizable": True,
    },
    {
        # Sana — linear flow DiT, Gemma-2, DC-AE 32× VAE. Diffusers repo, very fast.
        "id": "sana", "label": "Sana", "backend": "sana", "prediction": "flow",
        "resolution": 1024, "default_base": "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        "prompt_hint": "a photo of <token> person",
        "params_b": 1.6, "quantizable": True,
    },
    {
        # PixArt-Sigma — epsilon DiT, T5-only, KL VAE 4ch. Diffusers repo, lightweight.
        "id": "pixart", "label": "PixArt-Sigma", "backend": "pixart", "prediction": "epsilon",
        "resolution": 1024, "default_base": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "prompt_hint": "a photo of <token> person",
        "params_b": 0.6, "quantizable": True,
    },
    {
        # Bria 3.x — Flux-arch T5-only (no pooled/guidance), KL VAE 16ch. Diffusers repo.
        "id": "bria", "label": "Bria 3.2", "backend": "bria", "prediction": "flow",
        "resolution": 1024, "default_base": "briaai/BRIA-3.2",
        "prompt_hint": "a photo of <token> person",
        "params_b": 4.0, "quantizable": True,
    },
    {
        # AuraFlow — MMDiT flow, UMT5 text, KL VAE 4ch. Diffusers repo.
        "id": "auraflow", "label": "AuraFlow", "backend": "auraflow", "prediction": "flow",
        "resolution": 1024, "default_base": "fal/AuraFlow-v0.3",
        "prompt_hint": "a photo of <token> person",
        "params_b": 6.8, "quantizable": True,
    },
    {
        # CogView4 — flow DiT, GLM-4-9B text, KL VAE, SDXL micro-cond. Diffusers repo.
        "id": "cogview4", "label": "CogView4", "backend": "cogview4", "prediction": "flow",
        "resolution": 1024, "default_base": "THUDM/CogView4-6B",
        "prompt_hint": "a photo of <token> person",
        "params_b": 6.0, "quantizable": True,
    },
    {
        # Ovis-Image — Flux arch, Qwen3 text (chat template), no guidance. Diffusers repo.
        "id": "ovis", "label": "Ovis-Image", "backend": "ovis", "prediction": "flow",
        "resolution": 1024, "default_base": "",
        "prompt_hint": "a photo of <token> person",
        "params_b": 3.0, "quantizable": True,
    },
    {
        # Kolors — SDXL UNet + ChatGLM3-6B, epsilon. Diffusers repo.
        "id": "kolors", "label": "Kolors", "backend": "kolors", "prediction": "epsilon",
        "resolution": 1024, "default_base": "Kwai-Kolors/Kolors-diffusers",
        "prompt_hint": "a photo of <token> person",
        "params_b": 2.6, "quantizable": False,
    },
    {
        # HunyuanImage — flow DiT ~17B, Qwen2.5-VL (+ByT5 ignored), 32× VAE. Diffusers repo.
        "id": "hunyuanimage", "label": "HunyuanImage", "backend": "hunyuanimage", "prediction": "flow",
        "resolution": 1024, "default_base": "tencent/HunyuanImage-2.1",
        "prompt_hint": "a photo of <token> person",
        "params_b": 17.0, "quantizable": True,
    },
]

_BY_ID = {f["id"]: f for f in FAMILIES}


def get_family(arch: str) -> dict:
    """Return the family (fallback to SDXL if unknown)."""
    return _BY_ID.get((arch or "sdxl"), _BY_ID["sdxl"])


def soma_meta(cfg, family, step) -> dict:
    """Metadata embedded in the LoRA .safetensors (read back by /api/checkpoints to
    list by arch/name/date). Values = strings (safetensors constraint)."""
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
