"""Real LoRA training for SDXL — diffusers + peft (NOT kohya).

Isolated so torch/diffusers only load when simulate=False. The loop is deliberately
compact and readable: it's OUR engine, not a wrapper. To be refined on the first real
run (latent caching, bucketing, advanced optimizers will come next).
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
from train_utils import make_lr_scheduler, maybe_drop, min_snr_weights

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


# SDXL buckets ~1024² (multiples of 64): we keep the image ratio
# (portrait/landscape) instead of center-cropping to a square, which cropped the
# framing (body/underboob) of vertical images.
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
    """Load an image, pick the nearest ratio bucket, resize
    to cover then center-crop -> (PIL, W, H). No forced square."""
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
    directly in ComfyUI / A1111. `meta` = embedded SOMA metadata."""
    try:
        import torch
        from diffusers.utils import convert_state_dict_to_kohya
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        kohya = convert_state_dict_to_kohya(get_peft_model_state_dict(unet))
        # the PEFT wrapper leaves the "base_model_model_" prefix -> ComfyUI wants
        # "lora_unet_" for the UNet layers.
        kohya = {
            k.replace("base_model_model_", "lora_unet_"): v.detach().to("cpu", torch.float16)
            for k, v in kohya.items()
        }
        path = os.path.join(out_dir, name + ".safetensors")
        save_file(kohya, path, metadata=meta or None)
        emit(evt("log", level="info", message=f"ComfyUI LoRA exported: {name}.safetensors"))
        return path
    except Exception as e:
        emit(evt("log", level="warn", message=f"ComfyUI export failed: {e}"))
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

    # Training objective by family (SDXL/Pony/Illustrious = epsilon;
    # NoobAI v-pred = v_prediction + zero-terminal-SNR).
    family = family or {}
    prediction = family.get("prediction", "epsilon")
    zsnr = bool(family.get("zsnr", False))
    is_vpred = prediction == "v_prediction"

    base_model = clean_path(cfg.base_model)
    emit(evt("log", level="info", message=f"Loading {base_model} on {device}…"))
    # Checkpoint local (.safetensors/.ckpt) -> from_single_file ; sinon repo HF.
    if base_model.lower().endswith((".safetensors", ".ckpt")) and os.path.isfile(base_model):
        pipe = StableDiffusionXLPipeline.from_single_file(base_model, torch_dtype=dtype)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(base_model, torch_dtype=dtype)
    unet, vae = pipe.unet, pipe.vae
    te1, te2 = pipe.text_encoder, pipe.text_encoder_2
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
    # Training scheduler: we force prediction_type + zsnr by family.
    sched_kwargs = {}
    if is_vpred:
        sched_kwargs["prediction_type"] = "v_prediction"
    if zsnr:
        sched_kwargs["rescale_betas_zero_snr"] = True
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config, **sched_kwargs)
    if is_vpred:
        # the sampler must also be v-pred (else inconsistent previews)
        try:
            pipe.scheduler = pipe.scheduler.from_config(
                pipe.scheduler.config, prediction_type="v_prediction",
                rescale_betas_zero_snr=zsnr,
            )
        except Exception as e:
            emit(evt("log", level="warn", message=f"v-pred sampler not applied: {e}"))
    emit(evt("log", level="info",
             message=f"Objective: {prediction}{' + zsnr' if zsnr else ''}"))

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
    pipe.unet = unet  # so the samples reflect the LoRA
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]

    try:  # AdamW 8-bit if available (memory saving), otherwise standard AdamW
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(params, lr=cfg.learning_rate)
        emit(evt("log", level="info", message="Optimizer: AdamW8bit (bitsandbytes)"))
    except Exception:
        opt = torch.optim.AdamW(params, lr=cfg.learning_rate)
        emit(evt("log", level="info", message="Optimizer: AdamW (fallback)"))

    sched = make_lr_scheduler(opt, cfg.max_steps, getattr(cfg, "lr_warmup_ratio", 0.05))
    snr_gamma = getattr(cfg, "min_snr_gamma", 5.0)
    cap_drop = getattr(cfg, "caption_dropout", 0.0)

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"No images found in {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) in the dataset"))

    buckets = _buckets_for_resolution(cfg.resolution)
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    emit(evt("log", level="info", message="Ratio bucketing enabled (no square crop)"))

    def encode_prompt(caption, allow_drop=True):
        cap = caption or f"a photo of {cfg.instance_token} person"
        if allow_drop:
            cap = maybe_drop(cap, cap_drop)  # caption dropout -> unconditional prompt
        ids1 = tok1(cap, padding="max_length", max_length=tok1.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(device)
        ids2 = tok2(cap, padding="max_length", max_length=tok2.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            o1 = te1(ids1, output_hidden_states=True)
            o2 = te2(ids2, output_hidden_states=True)
        emb = torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1)
        return emb, o2[0]  # (encoder_hidden_states, pooled)

    def sample_loss(path, caption, allow_drop=True):
        """Weighted denoising loss for one image (no backward). Shared by the instance
        step and the prior-preservation step."""
        img, W, H = _load_bucketed(path, buckets)
        px = norm(img).unsqueeze(0).to(device, dtype=torch.float32)
        add_time_ids = torch.tensor([[H, W, 0, 0, H, W]], device=device, dtype=dtype)
        with torch.no_grad():
            latents = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
        latents = latents.to(dtype)
        noise = torch.randn_like(latents)
        ts = torch.randint(0, noise_sched.config.num_train_timesteps, (1,), device=device).long()
        noisy = noise_sched.add_noise(latents, noise, ts)
        emb, pooled = encode_prompt(caption, allow_drop)
        added = {"text_embeds": pooled, "time_ids": add_time_ids}
        pred = unet(noisy, ts, encoder_hidden_states=emb, added_cond_kwargs=added).sample
        target = noise_sched.get_velocity(latents, noise, ts) if is_vpred else noise  # v-pred | epsilon
        per = torch.nn.functional.mse_loss(
            pred.float(), target.float(), reduction="none"
        ).mean(dim=[1, 2, 3])
        w = min_snr_weights(noise_sched, ts, snr_gamma, prediction).to(per.device)
        return (per * w).mean()

    # Prior preservation (DreamBooth-style): if a folder of class/regularization images
    # is given, every step also denoises a class image so the LoRA keeps the base model's
    # prior instead of overwriting it. Empty -> off.
    reg_dir = clean_path(getattr(cfg, "reg_dataset_dir", ""))
    reg_data = _list_dataset(reg_dir) if reg_dir else []
    prior_w = float(getattr(cfg, "prior_loss_weight", 1.0))
    class_prompt = getattr(cfg, "class_prompt", "") or "a photo of a person"
    if reg_data:
        emit(evt("log", level="info",
                 message=f"Prior preservation: {len(reg_data)} reg image(s), weight {prior_w}"))

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
            loss = sample_loss(path, caption)
            if reg_data:  # anchor to the class prior
                rp, rc = random.choice(reg_data)
                loss = loss + prior_w * sample_loss(rp, rc or class_prompt, allow_drop=False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            sched.step()
            emit(
                evt("step", step=step, total_steps=cfg.max_steps,
                    loss=round(loss.item(), 4), lr=sched.get_last_lr()[0],
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
    # The VAE is fp32 (stable encoding during training) but the latents
    # produced by the UNet are bf16 -> mismatch at decode. We align the VAE
    # to the inference dtype for the sample, then restore fp32.
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
        emit(evt("log", level="warn", message=f"sample failed: {e}"))
    finally:
        pipe.vae.to(dtype=vae_dtype)
        unet.train()
        emit(evt("status", state="training", step=step))
