"""Real LoRA training for Stable Diffusion 1.5 — diffusers + peft (PAS kohya).

SD1.5 = UNet epsilon (comme SDXL) MAIS **un seul text encoder** (CLIP ViT-L, dim 768),
**no add_time_ids or pooled** (≠ SDXL). Native resolution 512. We reuse the
helpers de real_trainer (dataset, bucketing par ratio, export kohya `lora_unet_`).

forward = unet(noisy, timestep, encoder_hidden_states=clip[B,77,768]).sample ; cible =
noise (epsilon) or velocity (v-pred if the family declares it). LoRA export ComfyUI/kohya.
"""
import os
import random
import time

from captioner import clean_path
from events import evt
from train_utils import make_lr_scheduler, maybe_drop, min_snr_weights
# generic helpers shared with the SDXL trainer
from real_trainer import (
    _buckets_for_resolution, _export_comfyui_lora, _list_dataset, _load_bucketed,
)

_LORA_TARGETS = ["to_k", "to_q", "to_v", "to_out.0"]


def run_sd15_training(cfg, emit, stop_event, family=None):
    import base64
    import io

    import torch
    import torchvision.transforms as T
    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from peft import LoraConfig, get_peft_model

    from families import soma_meta

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
        getattr(cfg, "mixed_precision", "bf16"), torch.bfloat16
    )

    family = family or {}
    prediction = family.get("prediction", "epsilon")
    zsnr = bool(family.get("zsnr", False))
    is_vpred = prediction == "v_prediction"

    base_model = clean_path(cfg.base_model)
    emit(evt("log", level="info", message=f"Loading {base_model} on {device}…"))
    if base_model.lower().endswith((".safetensors", ".ckpt")) and os.path.isfile(base_model):
        pipe = StableDiffusionPipeline.from_single_file(base_model, torch_dtype=dtype, safety_checker=None)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype, safety_checker=None)
    unet, vae, te, tok = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer

    sched_kwargs = {}
    if is_vpred:
        sched_kwargs["prediction_type"] = "v_prediction"
    if zsnr:
        sched_kwargs["rescale_betas_zero_snr"] = True
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config, **sched_kwargs)
    emit(evt("log", level="info", message=f"Objective: {prediction}{' + zsnr' if zsnr else ''}"))

    for m in (vae, te, unet):
        m.requires_grad_(False)
    vae.to(device, dtype=torch.float32)
    te.to(device, dtype=dtype)
    unet.to(device, dtype=dtype)

    lora = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, init_lora_weights="gaussian",
                      target_modules=_LORA_TARGETS)
    unet = get_peft_model(unet, lora)
    pipe.unet = unet
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]

    try:
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(params, lr=cfg.learning_rate)
        emit(evt("log", level="info", message="Optimizer: AdamW8bit (bitsandbytes)"))
    except Exception:
        opt = torch.optim.AdamW(params, lr=cfg.learning_rate)
        emit(evt("log", level="info", message="Optimizer: AdamW (fallback)"))

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"No images found in {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) in the dataset"))

    buckets = _buckets_for_resolution(cfg.resolution)
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    def encode_prompt(caption):
        cap = caption or f"a photo of {cfg.instance_token} person"
        cap = maybe_drop(cap, getattr(cfg, "caption_dropout", 0.0))
        ids = tok(cap, padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            emb = te(ids)[0]  # last_hidden_state [1,77,768]
        return emb

    emit(evt("status", state="training", total_steps=cfg.max_steps))
    unet.train()
    sched = make_lr_scheduler(opt, cfg.max_steps, getattr(cfg, "lr_warmup_ratio", 0.05))
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
            with torch.no_grad():
                latents = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
            latents = latents.to(dtype)
            noise = torch.randn_like(latents)
            ts = torch.randint(0, noise_sched.config.num_train_timesteps, (1,), device=device).long()
            noisy = noise_sched.add_noise(latents, noise, ts)
            emb = encode_prompt(caption)
            pred = unet(noisy, ts, encoder_hidden_states=emb).sample
            target = noise_sched.get_velocity(latents, noise, ts) if is_vpred else noise
            per = torch.nn.functional.mse_loss(
                pred.float(), target.float(), reduction="none").mean(dim=[1, 2, 3])
            w = min_snr_weights(noise_sched, ts, getattr(cfg, "min_snr_gamma", 5.0), prediction).to(per.device)
            loss = (per * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            sched.step()
            emit(evt("step", step=step, total_steps=cfg.max_steps, loss=round(loss.item(), 4),
                     lr=sched.get_last_lr()[0], secs=round(time.time() - t0, 1)))
            if step % cfg.sample_every == 0 or step == cfg.max_steps:
                _sample(pipe, unet, cfg, step, emit, device)
        if stop_event.is_set():
            break

    out = os.path.join(cfg.output_dir, cfg.project_name)
    os.makedirs(out, exist_ok=True)
    unet.save_pretrained(out)
    comfy_path = _export_comfyui_lora(unet, out, cfg.project_name, emit,
                                      meta=soma_meta(cfg, family, step))
    emit(evt("status", state="done", step=step, secs=round(time.time() - t0, 1),
             output=out, comfyui=comfy_path))


def _sample(pipe, unet, cfg, step, emit, device):
    import base64
    import io

    import torch

    emit(evt("status", state="sampling", step=step))
    infer_dtype = next(unet.parameters()).dtype
    vae_dtype = pipe.vae.dtype
    try:
        unet.eval()
        pipe.vae.to(dtype=infer_dtype)
        with torch.no_grad():
            image = pipe(
                cfg.sample_prompt, num_inference_steps=24, guidance_scale=7.0,
                generator=torch.Generator(device=device).manual_seed(cfg.seed),
            ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        emit(evt("sample", step=step, total_steps=cfg.max_steps, placeholder=False,
                 image="data:image/png;base64," + b64, prompt=cfg.sample_prompt,
                 sharpness=round(step / cfg.max_steps, 3)))
    except Exception as e:
        emit(evt("log", level="warn", message=f"sample failed: {e}"))
    finally:
        pipe.vae.to(dtype=vae_dtype)
        unet.train()
        emit(evt("status", state="training", step=step))
