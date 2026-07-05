"""Vrai entraînement LoRA SDXL — diffusers + peft (PAS kohya).

Isolé pour ne charger torch/diffusers que quand simulate=False. Loop volontairement
compacte et lisible : c'est NOTRE moteur, pas un wrapper. À affiner au premier vrai
run (latent caching, bucketing, optimizers avancés viendront ensuite).
"""
import base64
import glob
import io
import os
import random
import time

from captioner import clean_path
from events import evt
from families import soma_meta

_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _list_dataset(d):
    pairs = []
    for p in sorted(glob.glob(os.path.join(d, "*"))):
        if not p.lower().endswith(_EXTS):
            continue
        cap_path = os.path.splitext(p)[0] + ".txt"
        caption = ""
        if os.path.exists(cap_path):
            with open(cap_path, encoding="utf-8") as f:
                caption = f.read().strip()
        pairs.append((p, caption))
    return pairs


# Buckets SDXL ~1024² (multiples de 64) : on garde le ratio des images
# (portrait/paysage) au lieu de center-cropper en carré, ce qui rognait le
# cadrage (corps/underboob) des images verticales.
_BASE_BUCKETS_1024 = [
    (1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216),
    (1344, 768), (768, 1344), (1408, 704), (704, 1408),
]


def _buckets_for_resolution(res):
    if int(res) == 1024:
        return _BASE_BUCKETS_1024
    s = float(res) / 1024.0
    return [
        (max(64, int(round(w * s / 64)) * 64), max(64, int(round(h * s / 64)) * 64))
        for w, h in _BASE_BUCKETS_1024
    ]


def _pick_bucket(w, h, buckets):
    ar = w / h
    return min(buckets, key=lambda b: abs((b[0] / b[1]) - ar))


def _load_bucketed(path, buckets):
    """Charge une image, choisit le bucket de ratio le plus proche, redimensionne
    pour couvrir puis center-crop -> (PIL, W, H). Pas de carré forcé."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    W, H = _pick_bucket(w, h, buckets)
    scale = max(W / w, H / h)
    nw, nh = max(W, round(w * scale)), max(H, round(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - W) // 2, (nh - H) // 2
    return img.crop((left, top, left + W, top + H)), W, H


def _export_comfyui_lora(unet, out_dir, name, emit, meta=None):
    """Convertit l'adaptateur PEFT en LoRA format kohya (.safetensors), chargeable
    directement dans ComfyUI / A1111. `meta` = métadonnées SOMA embarquées."""
    try:
        import torch
        from diffusers.utils import convert_state_dict_to_kohya
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        kohya = convert_state_dict_to_kohya(get_peft_model_state_dict(unet))
        # le wrapper PEFT laisse le préfixe "base_model_model_" -> ComfyUI veut
        # "lora_unet_" pour les couches de l'UNet.
        kohya = {
            k.replace("base_model_model_", "lora_unet_"): v.detach().to("cpu", torch.float16)
            for k, v in kohya.items()
        }
        path = os.path.join(out_dir, name + ".safetensors")
        save_file(kohya, path, metadata=meta or None)
        emit(evt("log", level="info", message=f"LoRA ComfyUI exporté: {name}.safetensors"))
        return path
    except Exception as e:
        emit(evt("log", level="warn", message=f"export ComfyUI échoué: {e}"))
        return None


def run_real_training(cfg, emit, stop_event, family=None):
    import torch
    import torchvision.transforms as T
    from diffusers import DDPMScheduler, StableDiffusionXLPipeline
    from peft import LoraConfig, get_peft_model
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(cfg.mixed_precision, torch.bfloat16)

    # Objectif d'entraînement selon la famille (SDXL/Pony/Illustrious = epsilon ;
    # NoobAI v-pred = v_prediction + zero-terminal-SNR).
    family = family or {}
    prediction = family.get("prediction", "epsilon")
    zsnr = bool(family.get("zsnr", False))
    is_vpred = prediction == "v_prediction"

    base_model = clean_path(cfg.base_model)
    emit(evt("log", level="info", message=f"Chargement {base_model} sur {device}…"))
    # Checkpoint local (.safetensors/.ckpt) -> from_single_file ; sinon repo HF.
    if base_model.lower().endswith((".safetensors", ".ckpt")) and os.path.isfile(base_model):
        pipe = StableDiffusionXLPipeline.from_single_file(base_model, torch_dtype=dtype)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(base_model, torch_dtype=dtype)
    unet, vae = pipe.unet, pipe.vae
    te1, te2 = pipe.text_encoder, pipe.text_encoder_2
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
    # Scheduler d'entraînement : on force prediction_type + zsnr selon la famille.
    sched_kwargs = {}
    if is_vpred:
        sched_kwargs["prediction_type"] = "v_prediction"
    if zsnr:
        sched_kwargs["rescale_betas_zero_snr"] = True
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config, **sched_kwargs)
    if is_vpred:
        # le sampler doit aussi être en v-pred (sinon aperçus incohérents)
        try:
            pipe.scheduler = pipe.scheduler.from_config(
                pipe.scheduler.config, prediction_type="v_prediction",
                rescale_betas_zero_snr=zsnr,
            )
        except Exception as e:
            emit(evt("log", level="warn", message=f"sampler v-pred non appliqué: {e}"))
    emit(evt("log", level="info",
             message=f"Objectif: {prediction}{' + zsnr' if zsnr else ''}"))

    for m in (vae, te1, te2, unet):
        m.requires_grad_(False)
    vae.to(device, dtype=torch.float32)  # VAE en fp32 = plus stable
    te1.to(device, dtype=dtype)
    te2.to(device, dtype=dtype)
    unet.to(device, dtype=dtype)

    lora = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora)
    pipe.unet = unet  # pour que les samples reflètent le LoRA
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]

    try:  # AdamW 8-bit si dispo (gain mémoire), sinon AdamW standard
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(params, lr=cfg.learning_rate)
        emit(evt("log", level="info", message="Optimizer: AdamW8bit (bitsandbytes)"))
    except Exception:
        opt = torch.optim.AdamW(params, lr=cfg.learning_rate)
        emit(evt("log", level="info", message="Optimizer: AdamW (fallback)"))

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"Aucune image trouvée dans {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) dans le dataset"))

    buckets = _buckets_for_resolution(cfg.resolution)
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    emit(evt("log", level="info", message="Bucketing par ratio activé (pas de crop carré)"))

    def encode_prompt(caption):
        cap = caption or f"a photo of {cfg.instance_token} person"
        ids1 = tok1(cap, padding="max_length", max_length=tok1.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(device)
        ids2 = tok2(cap, padding="max_length", max_length=tok2.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            o1 = te1(ids1, output_hidden_states=True)
            o2 = te2(ids2, output_hidden_states=True)
        emb = torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1)
        return emb, o2[0]  # (encoder_hidden_states, pooled)

    emit(evt("status", state="training", total_steps=cfg.max_steps))
    unet.train()
    t0 = time.time()
    step = 0
    while step < cfg.max_steps:
        random.shuffle(data)
        for path, caption in data:
            if stop_event.is_set() or step >= cfg.max_steps:
                break
            step += 1
            img, W, H = _load_bucketed(path, buckets)
            px = norm(img).unsqueeze(0).to(device, dtype=torch.float32)
            add_time_ids = torch.tensor(
                [[H, W, 0, 0, H, W]], device=device, dtype=dtype
            )
            with torch.no_grad():
                latents = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
            latents = latents.to(dtype)
            noise = torch.randn_like(latents)
            ts = torch.randint(
                0, noise_sched.config.num_train_timesteps, (1,), device=device
            ).long()
            noisy = noise_sched.add_noise(latents, noise, ts)
            emb, pooled = encode_prompt(caption)
            added = {"text_embeds": pooled, "time_ids": add_time_ids}
            pred = unet(noisy, ts, encoder_hidden_states=emb,
                        added_cond_kwargs=added).sample
            # Cible : velocity (v-pred) ou bruit (epsilon)
            target = noise_sched.get_velocity(latents, noise, ts) if is_vpred else noise
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
                _sample(pipe, unet, cfg, step, emit, device)
        if stop_event.is_set():
            break

    out = os.path.join(cfg.output_dir, cfg.project_name)
    os.makedirs(out, exist_ok=True)
    unet.save_pretrained(out)  # format PEFT (rechargeable)
    comfy_path = _export_comfyui_lora(unet, out, cfg.project_name, emit, meta=soma_meta(cfg, family, step))
    emit(evt("status", state="done", step=step,
             secs=round(time.time() - t0, 1), output=out, comfyui=comfy_path))


def _sample(pipe, unet, cfg, step, emit, device):
    import torch

    emit(evt("status", state="sampling", step=step))
    # Le VAE est en fp32 (encodage stable à l'entraînement) mais les latents
    # produits par l'UNet sont en bf16 -> mismatch au décodage. On aligne le VAE
    # sur le dtype d'inférence le temps du sample, puis on restaure fp32.
    infer_dtype = next(unet.parameters()).dtype
    vae_dtype = pipe.vae.dtype
    try:
        unet.eval()
        pipe.vae.to(dtype=infer_dtype)
        with torch.no_grad():
            image = pipe(
                cfg.sample_prompt,
                num_inference_steps=24,
                guidance_scale=5.0,
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
        pipe.vae.to(dtype=vae_dtype)
        unet.train()
        emit(evt("status", state="training", step=step))
