"""Vrai entraînement LoRA Z-Image Turbo — diffusers ZImagePipeline + peft (PAS Ostris).

Z-Image = DiT single-stream 6B, objectif flow-matching (rectified flow), pas epsilon.
Composants : VAE (AutoencoderKL, latents 16 canaux avec shift_factor), text encoder
Qwen (chat template, on prend hidden_states[-2], dim 2560), transformer 6B.

⚠️ Contrainte 16 Go : impossible de tenir VAE + Qwen + transformer 6B EN MÊME TEMPS
avec les gradients. Stratégie = on PRÉ-CALCULE une fois les latents VAE et les
embeddings texte (+ le prompt d'aperçu), on décharge le text encoder et on ne garde
que le transformer sur le GPU pour la boucle d'entraînement. Le VAE reste au chaud
(CPU) et remonte sur le GPU juste le temps d'un aperçu.

Détails d'objectif (vérifiés sur la pipeline diffusers) :
- latent "modèle" x1 = (vae.encode(img).sample() - shift_factor) * scaling_factor
- bruit x0 ~ N(0,1) ; sigma ∈ (0,1) (logit-normal)
- latent bruité = sigma*x0 + (1-sigma)*x1   (sigma=1 -> bruit pur, sigma=0 -> data)
- timestep passé au modèle = 1 - sigma  (la pipeline utilise (1000 - t)/1000)
- la pipeline NÉGE la sortie du transformer avant le scheduler => le modèle prédit
  nativement (data - noise). Donc cible d'entraînement = x1 - x0.
"""
import base64
import gc
import io
import math
import os
import random
import time

from captioner import clean_path
from events import evt

# Helpers génériques réutilisés du trainer SDXL (dataset + bucketing par ratio).
# Les buckets sont des multiples de 64 -> compatibles Z-Image (exige multiples de 16).
from real_trainer import _buckets_for_resolution, _list_dataset, _load_bucketed

ZIMAGE_DEFAULT = "Tongyi-MAI/Z-Image-Turbo"
QWEN_TE_REPO = "Qwen/Qwen3-4B"  # config + tokenizer seulement (quelques Mo), PAS les poids

# LoRA sur les projections d'attention + FFN des blocs du transformer (pas les
# embedders/adaLN/final_layer, qui ne matchent pas ces suffixes).
_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]


def _find_models_root(dit_path):
    """Remonte l'arbo depuis le DiT jusqu'à un dossier ComfyUI `models`
    (contenant vae/ et text_encoders/)."""
    d = os.path.dirname(os.path.abspath(dit_path))
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "vae")) and os.path.isdir(os.path.join(d, "text_encoders")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _auto_component(root, subdir, preferred, patterns):
    """Trouve un composant (vae/text_encoder) dans models/<subdir> : nom préféré
    d'abord, sinon 1er .safetensors matchant un motif (évite gguf/fp8)."""
    import glob as _glob

    folder = os.path.join(root, subdir)
    pref = os.path.join(folder, preferred)
    if os.path.isfile(pref):
        return pref
    cands = sorted(_glob.glob(os.path.join(folder, "*.safetensors")))
    for pat in patterns:
        for c in cands:
            name = os.path.basename(c).lower()
            if pat in name and "fp8" not in name and "scaled" not in name:
                return c
    return None


def _build_pipeline_from_files(dit_path, vae_path, te_path, dtype, emit, need_te=True, precision="bf16"):
    """Assemble ZImagePipeline depuis 3 fichiers ComfyUI séparés : DiT Z-Image,
    VAE Flux (ae.zimage), text encoder Qwen3-4B. Ne télécharge QUE de petits
    configs/tokenizer (pas les poids lourds, on a tout en local).
    need_te=False (cache d'embeddings présent) -> on NE CHARGE PAS Qwen (8 Go).
    precision int8/nf4 -> quantifie le DiT (moins de VRAM, sur petites cartes)."""
    import safetensors.torch as st
    import torch
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        ZImagePipeline,
        ZImageTransformer2DModel,
    )
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from quant import bnb_config, patch_single_file_fresh_quant

    emit(evt("log", level="info", message=f"DiT (single-file): {os.path.basename(dit_path)}"))
    bnb = bnb_config(precision)
    if bnb is not None:
        patch_single_file_fresh_quant()
        emit(evt("log", level="info", message=f"Quantization DiT: {precision}"))
        transformer = ZImageTransformer2DModel.from_single_file(
            dit_path, quantization_config=bnb, torch_dtype=dtype
        )
    else:
        transformer = ZImageTransformer2DModel.from_single_file(dit_path, torch_dtype=dtype)

    emit(evt("log", level="info", message=f"VAE (Flux AE): {os.path.basename(vae_path)}"))
    # ⚠️ sans config explicite, from_single_file instancie un VAE SD 4 canaux ->
    # mismatch (le VAE Z-Image/Flux fait 16 canaux). On force la config du repo.
    vae = AutoencoderKL.from_single_file(
        vae_path, config=ZIMAGE_DEFAULT, subfolder="vae", torch_dtype=dtype
    )

    if need_te:
        emit(evt("log", level="info", message=f"Text encoder Qwen3: {os.path.basename(te_path)}"))
        from accelerate import init_empty_weights

        te_cfg = AutoConfig.from_pretrained(QWEN_TE_REPO)
        # init sur meta device (instantané, pas d'alloc fp32 16 Go ni d'init aléatoire) ;
        # include_buffers=False -> les buffers (rotary inv_freq) restent réels.
        with init_empty_weights(include_buffers=False):
            te = AutoModel.from_config(te_cfg)  # Qwen3Model
        sd = st.load_file(te_path)
        sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
        sd.pop("lm_head.weight", None)
        # assign=True : place directement les tenseurs bf16 du disque (pas de copie fp32)
        missing, unexpected = te.load_state_dict(sd, strict=False, assign=True)
        real_missing = [k for k in missing if "rotary" not in k and "inv_freq" not in k]
        if real_missing or unexpected:
            emit(evt("log", level="warn",
                     message=f"Qwen load: {len(real_missing)} manquants, {len(unexpected)} inattendus"))
        tokenizer = AutoTokenizer.from_pretrained(QWEN_TE_REPO)
    else:
        emit(evt("log", level="info", message="Cache d'embeddings présent → Qwen non chargé (gain de temps)"))
        te = None
        tokenizer = None

    scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
    )
    return ZImagePipeline(
        scheduler=scheduler, vae=vae, text_encoder=te, tokenizer=tokenizer, transformer=transformer
    )


def _load_pipeline(cfg, dtype, emit, need_te=True):
    """Charge la pipeline Z-Image. Priorité : fichiers ComfyUI locaux (0 download
    lourd) si base_model pointe un DiT .safetensors ; sinon repo diffusers HF.
    need_te=False -> on saute le chargement du text encoder Qwen (cache présent)."""
    base = clean_path(cfg.base_model) or ZIMAGE_DEFAULT
    is_local = base.lower().endswith((".safetensors", ".ckpt")) and os.path.isfile(base)
    if not is_local:
        from diffusers import ZImagePipeline

        emit(evt("log", level="info", message=f"Z-Image : from_pretrained {base} (repo HF)…"))
        pipe = ZImagePipeline.from_pretrained(base, torch_dtype=dtype)
        if not need_te and getattr(pipe, "text_encoder", None) is not None:
            del pipe.text_encoder
            pipe.text_encoder = None
        return pipe

    # Fichiers séparés : localise VAE + text encoder
    vae_path = clean_path(cfg.zimage_vae)
    te_path = clean_path(cfg.zimage_text_encoder)
    if not vae_path or not te_path:
        root = _find_models_root(base)
        if root is None:
            raise RuntimeError(
                "DiT local détecté mais impossible de trouver l'arbo ComfyUI models/ "
                "(vae, text_encoders). Renseigne zimage_vae et zimage_text_encoder."
            )
        vae_path = vae_path or _auto_component(root, "vae", "ae.zimage.safetensors", ["zimage", "ae"])
        te_path = te_path or _auto_component(root, "text_encoders", "qwen_3_4b.safetensors", ["qwen_3", "qwen3"])
    if not vae_path or not os.path.isfile(vae_path):
        raise RuntimeError(f"VAE Z-Image introuvable (cherché ae.zimage). Donné: {vae_path!r}")
    if need_te and (not te_path or not os.path.isfile(te_path)):
        raise RuntimeError(f"Text encoder Qwen3 introuvable. Donné: {te_path!r}")
    return _build_pipeline_from_files(
        base, vae_path, te_path, dtype, emit, need_te=need_te,
        precision=getattr(cfg, "precision", "bf16"),
    )


def _cache_key(data, cfg):
    """Empreinte du dataset (chemins + mtime + taille + caption) + résolution +
    token + prompt d'aperçu. Toute modif d'image ou de .txt invalide le cache."""
    import hashlib

    h = hashlib.sha1()
    h.update(f"{cfg.resolution}|{cfg.instance_token}|{cfg.sample_prompt}".encode("utf-8", "ignore"))
    for path, cap in data:
        try:
            st_ = os.stat(path)
            sig = f"{path}|{int(st_.st_mtime)}|{st_.st_size}|{cap}"
        except OSError:
            sig = f"{path}|0|0|{cap}"
        h.update(sig.encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def _load_cache(cache_file, emit):
    import torch

    try:
        blob = torch.load(cache_file, map_location="cpu")
        return blob["latents"], blob["embeds"], blob["sample"]
    except Exception as e:
        emit(evt("log", level="warn", message=f"cache illisible ({e}) → recalcul"))
        return None


def _save_cache(cache_file, latents_cache, emb_cache, sample_emb, emit):
    import torch

    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save({"latents": latents_cache, "embeds": emb_cache, "sample": sample_emb}, cache_file)
        emit(evt("log", level="info", message="Cache latents+texte enregistré (runs suivants plus rapides)"))
    except Exception as e:
        emit(evt("log", level="warn", message=f"écriture cache échouée: {e}"))


def _sample_sigma(device):
    """Logit-normal : concentre l'échantillonnage vers le milieu du trajet (mieux
    que l'uniforme pour le rectified flow)."""
    import torch

    u = torch.randn(1, device=device)
    return torch.sigmoid(u).item()


def _export_lora(transformer, out_dir, name, emit, meta=None):
    """Exporte l'adaptateur LoRA au format diffusers (préfixe `transformer.`),
    chargeable par ZImagePipeline.load_lora_weights et les loaders Z-Image récents."""
    try:
        import torch
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        sd = get_peft_model_state_dict(transformer)
        out = {}
        for k, v in sd.items():
            # peft -> diffusers : `base_model.model.<...>` devient `transformer.<...>`
            k = k.replace("base_model.model.", "")
            if not k.startswith("transformer."):
                k = "transformer." + k
            out[k] = v.detach().to("cpu", torch.float16)
        path = os.path.join(out_dir, name + ".safetensors")
        save_file(out, path, metadata=meta or None)
        emit(evt("log", level="info", message=f"LoRA Z-Image exporté: {name}.safetensors"))
        return path
    except Exception as e:
        emit(evt("log", level="warn", message=f"export LoRA échoué: {e}"))
        return None


def run_zimage_training(cfg, emit, stop_event):
    import torch
    import torchvision.transforms as T
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
        cfg.mixed_precision, torch.bfloat16
    )

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"Aucune image trouvée dans {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) dans le dataset"))

    buckets = _buckets_for_resolution(cfg.resolution)
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])  # -> [-1, 1]

    # Cache persistant : si les latents + embeddings de CE dataset/résolution/token
    # existent déjà sur disque, on saute l'encodage ET le chargement de Qwen (8 Go).
    cache_file = os.path.join(cfg.output_dir, ".soma_cache", f"zimage_{_cache_key(data, cfg)}.pt")
    cached = _load_cache(cache_file, emit) if os.path.isfile(cache_file) else None

    pipe = _load_pipeline(cfg, dtype, emit, need_te=(cached is None))

    if cached is not None:
        emit(evt("log", level="info", message="Cache trouvé → latents + embeddings réutilisés"))
        latents_cache, emb_cache, sample_emb = cached
        if getattr(pipe, "vae", None) is not None:
            pipe.vae.to("cpu", dtype=dtype)  # gardé au chaud pour l'aperçu
    else:
        # 1) latents VAE (le VAE reste ensuite au chaud sur CPU pour l'aperçu)
        emit(evt("log", level="info", message="Pré-calcul des latents VAE…"))
        vae = pipe.vae
        scaling = vae.config.scaling_factor
        shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
        vae.to(device, dtype=torch.float32)  # encodage en fp32 = stable
        latents_cache = []  # (latent_cpu, caption)
        with torch.no_grad():
            for path, caption in data:
                img, W, H = _load_bucketed(path, buckets)
                px = norm(img).unsqueeze(0).to(device, dtype=torch.float32)
                raw = vae.encode(px).latent_dist.sample()
                x1 = (raw - shift) * scaling
                latents_cache.append((x1.squeeze(0).to("cpu", torch.float32), caption))
        vae.to("cpu", dtype=dtype)  # libère le GPU, garde le VAE pour l'aperçu
        gc.collect()
        torch.cuda.empty_cache()

        # 2) embeddings texte (Qwen), puis on DÉCHARGE le text encoder
        emit(evt("log", level="info", message="Pré-calcul des embeddings texte (Qwen)…"))
        pipe.text_encoder.to(device)
        emb_cache = []
        default_cap = f"a photo of {cfg.instance_token} person"
        with torch.no_grad():
            for _, caption in data:
                pe, _ = pipe.encode_prompt(
                    caption or default_cap, device=device, do_classifier_free_guidance=False
                )
                emb_cache.append(pe[0].to("cpu", dtype))
            # embedding du prompt d'aperçu (permet de sampler SANS recharger Qwen)
            sample_pe, _ = pipe.encode_prompt(
                cfg.sample_prompt, device=device, do_classifier_free_guidance=False
            )
            sample_emb = sample_pe[0].to("cpu", dtype)
        pipe.text_encoder.to("cpu")
        del pipe.text_encoder
        pipe.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()

        _save_cache(cache_file, latents_cache, emb_cache, sample_emb, emit)

    # ------------------------------------------------------------------
    # 3) TRANSFORMER + LoRA sur le GPU (seul modèle résident pendant la boucle)
    # ------------------------------------------------------------------
    emit(evt("log", level="info", message="Transformer 6B -> GPU + LoRA…"))
    transformer = pipe.transformer
    from quant import is_quantized

    if is_quantized(getattr(cfg, "precision", "bf16")):
        transformer.to(device)  # déclenche la vraie quantization nf4/int8 sur GPU
    else:
        transformer.to(device, dtype=dtype)
    transformer.requires_grad_(False)
    lora = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        init_lora_weights="gaussian",
        target_modules=_LORA_TARGETS,
    )
    transformer = get_peft_model(transformer, lora)
    pipe.transformer = transformer
    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    params = [p for p in transformer.parameters() if p.requires_grad]

    try:
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(params, lr=cfg.learning_rate)
        emit(evt("log", level="info", message="Optimizer: AdamW8bit (bitsandbytes)"))
    except Exception:
        opt = torch.optim.AdamW(params, lr=cfg.learning_rate)
        emit(evt("log", level="info", message="Optimizer: AdamW (fallback)"))

    emit(evt("status", state="training", total_steps=cfg.max_steps))
    transformer.train()
    t0 = time.time()
    step = 0
    idx = list(range(len(latents_cache)))
    while step < cfg.max_steps:
        random.shuffle(idx)
        for i in idx:
            if stop_event.is_set() or step >= cfg.max_steps:
                break
            step += 1
            x1 = latents_cache[i][0].unsqueeze(0).to(device, dtype)  # [1,16,h,w]
            cap_emb = emb_cache[i].to(device, dtype)                 # [seq,2560]
            x0 = torch.randn_like(x1)
            sigma = _sample_sigma(device)
            noisy = sigma * x0 + (1.0 - sigma) * x1
            model_t = torch.tensor([1.0 - sigma], device=device, dtype=dtype)

            # x = liste de tenseurs (C, F=1, H, W) ; cap_feats = liste [seq, 2560]
            x_list = list(noisy.unsqueeze(2).unbind(dim=0))
            pred = transformer(x_list, model_t, [cap_emb], return_dict=False)[0]
            pred = torch.stack(pred, dim=0).squeeze(2)  # [1,16,h,w]

            target = x1 - x0  # le modèle prédit nativement (data - noise)
            loss = torch.nn.functional.mse_loss(pred.float(), target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()

            emit(
                evt("step", step=step, total_steps=cfg.max_steps,
                    loss=round(loss.item(), 4), lr=cfg.learning_rate,
                    secs=round(time.time() - t0, 1))
            )
            if step % cfg.sample_every == 0 or step == cfg.max_steps:
                _sample(pipe, transformer, sample_emb, cfg, step, emit, device, dtype)
        if stop_event.is_set():
            break

    out = os.path.join(cfg.output_dir, cfg.project_name)
    os.makedirs(out, exist_ok=True)
    transformer.save_pretrained(out)  # format PEFT (rechargeable)
    from families import get_family, soma_meta

    lora_path = _export_lora(transformer, out, cfg.project_name, emit,
                             meta=soma_meta(cfg, get_family(getattr(cfg, "arch", "zimage")), step))
    emit(evt("status", state="done", step=step,
             secs=round(time.time() - t0, 1), output=out, comfyui=lora_path))


def _sample(pipe, transformer, sample_emb, cfg, step, emit, device, dtype):
    """Aperçu Turbo (peu de steps, guidance 0). On remonte le VAE sur le GPU et on
    passe l'embedding du prompt DÉJÀ calculé -> pas besoin de recharger Qwen."""
    import torch

    emit(evt("status", state="sampling", step=step))
    try:
        transformer.eval()
        pipe.vae.to(device, dtype=dtype)
        with torch.no_grad():
            image = pipe(
                prompt=None,
                prompt_embeds=[sample_emb.to(device, dtype)],
                height=cfg.resolution,
                width=cfg.resolution,
                num_inference_steps=9,
                guidance_scale=0.0,  # Turbo : pas de CFG
                generator=torch.Generator(device=device).manual_seed(cfg.seed),
            ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        emit(
            evt("sample", step=step, total_steps=cfg.max_steps, placeholder=False,
                image="data:image/png;base64," + b64, prompt=cfg.sample_prompt,
                sharpness=round(step / cfg.max_steps, 3))
        )
    except Exception as e:
        emit(evt("log", level="warn", message=f"sample échoué: {e}"))
    finally:
        pipe.vae.to("cpu")
        transformer.train()
        torch.cuda.empty_cache()
        emit(evt("status", state="training", step=step))
