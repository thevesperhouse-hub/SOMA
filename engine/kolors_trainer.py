"""Vrai entraînement LoRA Kolors — diffusers + peft.

Kolors = UNet **SDXL** (epsilon, add_time_ids) mais texte encodé par **ChatGLM3-6B**
(au lieu du double CLIP). VAE SDXL 4 canaux. Distribué en repo diffusers →
from_pretrained(subfolder). base_model défaut Kwai-Kolors/Kolors-diffusers.

Vérifié (pipeline_kolors.py) : ChatGLM(ids, mask, position_ids, output_hidden_states) →
encoder_hidden_states = `hidden_states[-2].permute(1,0,2)` ; pooled = `hidden_states[-1][-1,:,:]`.
UNet forward = unet(noisy, ts, encoder_hidden_states, added_cond_kwargs={text_embeds:pooled,
time_ids:add_time_ids}). Entraînement DDPM epsilon : ts~U[0,1000], cible = bruit.
Export LoRA kohya `lora_unet_` (compatible ComfyUI).
"""
import os
import random
import time

from captioner import clean_path
from events import evt
from real_trainer import _buckets_for_resolution, _export_comfyui_lora, _list_dataset, _load_bucketed

_KOLORS_REPO = "Kwai-Kolors/Kolors-diffusers"
_TOK_MAX = 256
_LORA_TARGETS = ["to_k", "to_q", "to_v", "to_out.0"]


def _resolve(base_model):
    b = clean_path(base_model)
    return b if b else _KOLORS_REPO


def run_kolors_training(cfg, emit, stop_event, family=None):
    import torch
    import torchvision.transforms as T
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.pipelines.kolors.text_encoder import ChatGLMModel
    from diffusers.pipelines.kolors.tokenizer import ChatGLMTokenizer
    from peft import LoraConfig, get_peft_model

    from families import soma_meta

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16  # ChatGLM3-6B natif fp16
    src = _resolve(cfg.base_model)

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"Aucune image dans {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) — Kolors LoRA depuis {src}"))

    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    buckets = _buckets_for_resolution(cfg.resolution)
    noise_sched = DDPMScheduler.from_pretrained(src, subfolder="scheduler")

    # ---------------- 1) cache latents (VAE SDXL 4ch) ----------------
    emit(evt("log", level="info", message="Pré-calcul des latents (VAE)…"))
    vae = AutoencoderKL.from_pretrained(src, subfolder="vae", torch_dtype=torch.float32).to(device)
    scaling = vae.config.scaling_factor
    latents_cache = []
    with torch.no_grad():
        for path, caption in data:
            img, W, H = _load_bucketed(path, buckets)
            px = norm(img).unsqueeze(0).to(device, torch.float32)
            lat = vae.encode(px).latent_dist.sample() * scaling  # [1,4,h,w]
            latents_cache.append((lat.squeeze(0).to("cpu", dtype), caption, (H, W)))
    del vae
    import gc; gc.collect(); torch.cuda.empty_cache()

    # ---------------- 2) cache embeddings texte (ChatGLM) ----------------
    emit(evt("log", level="info", message="Pré-calcul des embeddings texte (ChatGLM3-6B)…"))
    tok = ChatGLMTokenizer.from_pretrained(src, subfolder="tokenizer")
    te = ChatGLMModel.from_pretrained(src, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
    default_cap = f"a photo of {cfg.instance_token} person"
    emb_cache = []
    with torch.no_grad():
        for _, caption, _hw in latents_cache:
            ti = tok(caption or default_cap, padding="max_length", max_length=_TOK_MAX,
                     truncation=True, return_tensors="pt").to(device)
            out = te(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"],
                     position_ids=ti["position_ids"], output_hidden_states=True)
            emb = out.hidden_states[-2].permute(1, 0, 2).clone()      # [1,seq,hidden]
            pooled = out.hidden_states[-1][-1, :, :].clone()          # [1,hidden]
            emb_cache.append((emb.to("cpu", dtype), pooled.to("cpu", dtype)))
    del te
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 3) UNet SDXL + LoRA ----------------
    emit(evt("log", level="info", message="Chargement UNet Kolors…"))
    unet = UNet2DConditionModel.from_pretrained(src, subfolder="unet", torch_dtype=dtype).to(device)
    unet.requires_grad_(False)
    lora = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, init_lora_weights="gaussian",
                      target_modules=_LORA_TARGETS)
    unet = get_peft_model(unet, lora)
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(params, lr=cfg.learning_rate)
        emit(evt("log", level="info", message="Optimizer: AdamW8bit"))
    except Exception:
        opt = torch.optim.AdamW(params, lr=cfg.learning_rate)

    emit(evt("status", state="training", total_steps=cfg.max_steps))
    unet.train()
    t0 = time.time()
    step = 0
    idx = list(range(len(latents_cache)))
    while step < cfg.max_steps:
        random.shuffle(idx)
        for i in idx:
            if stop_event.is_set() or step >= cfg.max_steps:
                break
            step += 1
            x1 = latents_cache[i][0].unsqueeze(0).to(device, dtype)  # [1,4,h,w]
            H, W = latents_cache[i][2]
            emb, pooled = emb_cache[i]
            emb = emb.to(device, dtype)
            pooled = pooled.to(device, dtype)
            add_time_ids = torch.tensor([[H, W, 0, 0, H, W]], device=device, dtype=dtype)
            noise = torch.randn_like(x1)
            ts = torch.randint(0, noise_sched.config.num_train_timesteps, (1,), device=device).long()
            noisy = noise_sched.add_noise(x1, noise, ts)
            added = {"text_embeds": pooled, "time_ids": add_time_ids}
            pred = unet(noisy, ts, encoder_hidden_states=emb, added_cond_kwargs=added).sample
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float())  # epsilon
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()

            emit(evt("step", step=step, total_steps=cfg.max_steps, loss=round(loss.item(), 4),
                     lr=cfg.learning_rate, secs=round(time.time() - t0, 1)))
        if stop_event.is_set():
            break

    out = os.path.join(cfg.output_dir, cfg.project_name)
    os.makedirs(out, exist_ok=True)
    unet.save_pretrained(out)
    comfy_path = _export_comfyui_lora(unet, out, cfg.project_name, emit,
                                      meta=soma_meta(cfg, family or {}, step))
    emit(evt("status", state="done", step=step, secs=round(time.time() - t0, 1),
             output=out, comfyui=comfy_path))
