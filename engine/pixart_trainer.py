"""Real PixArt-Sigma LoRA training — diffusers + peft.

PixArt-Sigma = **epsilon** DiT (noise, NOT flow), **T5-XXL** text only, AutoencoderKL VAE
4 channels (SD VAE). Micro-conditioning disabled (Sigma) → neutral added_cond_kwargs.
learned-sigma: the model outputs 2× the channels (noise+variance) → keep the noise half.
Distributed as a diffusers repo → from_pretrained. base_model default PixArt-alpha/PixArt-Sigma-XL-2-1024-MS.

forward = transformer(hidden_states[B,4,H,W], encoder_hidden_states=T5[B,seq,4096],
encoder_attention_mask, timestep=ts_discrete, added_cond_kwargs). DDPM-style training:
ts ~ U[0,1000], noisy = add_noise, TARGET = noise (epsilon).
"""
import gc
import os
import random
import time

from captioner import clean_path
from events import evt
from flux_trainer import _export_lora, _load_square
from real_trainer import _list_dataset

_PIXART_REPO = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
_T5_MAX = 300
_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


def _resolve(base_model):
    b = clean_path(base_model)
    return b if b else _PIXART_REPO


def run_pixart_training(cfg, emit, stop_event, family=None):
    import torch
    import torchvision.transforms as T
    from diffusers import AutoencoderKL, DDPMScheduler, PixArtTransformer2DModel
    from peft import LoraConfig, get_peft_model
    from transformers import T5EncoderModel, T5Tokenizer

    from quant import bnb_config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    precision = getattr(cfg, "precision", "bf16") or "bf16"
    src = _resolve(cfg.base_model)

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"No images in {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) — PixArt-Σ LoRA ({precision}) from {src}"))

    res = int(cfg.resolution)
    if res % 8 != 0:
        res = (res // 8) * 8
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    noise_sched = DDPMScheduler.from_pretrained(src, subfolder="scheduler")

    # ---------------- 1) cache latents (VAE KL 4ch) ----------------
    emit(evt("log", level="info", message="Pre-computing latents (VAE)…"))
    vae = AutoencoderKL.from_pretrained(src, subfolder="vae", torch_dtype=torch.float32).to(device)
    scaling = vae.config.scaling_factor
    latents_cache = []
    with torch.no_grad():
        for path, caption in data:
            px = norm(_load_square(path, res)).unsqueeze(0).to(device, torch.float32)
            x1 = vae.encode(px).latent_dist.sample() * scaling  # [1,4,h,w]
            latents_cache.append((x1.squeeze(0).to("cpu", dtype), caption))
    del vae
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 2) cache embeddings texte (T5) ----------------
    emit(evt("log", level="info", message="Pre-computing text embeddings (T5)…"))
    tok = T5Tokenizer.from_pretrained(src, subfolder="tokenizer")
    te = T5EncoderModel.from_pretrained(src, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
    default_cap = f"a photo of {cfg.instance_token} person"
    emb_cache = []
    with torch.no_grad():
        for _, caption in data:
            toks = tok(caption or default_cap, padding="max_length", max_length=_T5_MAX,
                       truncation=True, return_tensors="pt").to(device)
            emb = te(toks.input_ids, attention_mask=toks.attention_mask)[0]
            emb_cache.append((emb.to("cpu", dtype), toks.attention_mask.to("cpu")))
    del te
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 3) transformer + LoRA ----------------
    emit(evt("log", level="info", message=f"DiT PixArt-Σ ({precision})…"))
    bnb = bnb_config(precision)
    tkw = dict(subfolder="transformer", torch_dtype=dtype)
    if bnb is not None:
        tkw["quantization_config"] = bnb
        tkw["device_map"] = {"": 0}  # nf4 quant on GPU
    transformer = PixArtTransformer2DModel.from_pretrained(src, **tkw)
    if bnb is None:
        transformer = transformer.to(device)
    latent_ch = transformer.config.in_channels
    transformer.requires_grad_(False)
    lora = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, init_lora_weights="gaussian",
                      target_modules=_LORA_TARGETS)
    transformer = get_peft_model(transformer, lora)
    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    params = [p for p in transformer.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb2

        opt = bnb2.optim.AdamW8bit(params, lr=cfg.learning_rate)
        emit(evt("log", level="info", message="Optimizer: AdamW8bit"))
    except Exception:
        opt = torch.optim.AdamW(params, lr=cfg.learning_rate)

    added = {"resolution": None, "aspect_ratio": None}  # Sigma: micro-cond disabled
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
            x1 = latents_cache[i][0].unsqueeze(0).to(device, dtype)  # [1,4,h,w]
            emb, mask = emb_cache[i]
            emb = emb.to(device, dtype)
            mask = mask.to(device)
            noise = torch.randn_like(x1)
            ts = torch.randint(0, noise_sched.config.num_train_timesteps, (1,), device=device).long()
            noisy = noise_sched.add_noise(x1, noise, ts)

            pred = transformer(
                hidden_states=noisy, encoder_hidden_states=emb,
                encoder_attention_mask=mask, timestep=ts, added_cond_kwargs=added,
                return_dict=False,
            )[0]
            if pred.shape[1] == 2 * latent_ch:  # learned sigma -> keep the noise half
                pred = pred[:, :latent_ch]
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
    transformer.save_pretrained(out)
    from families import get_family, soma_meta

    lora_path = _export_lora(transformer, out, cfg.project_name, emit,
                             meta=soma_meta(cfg, get_family(getattr(cfg, "arch", "pixart")), step))
    emit(evt("status", state="done", step=step, secs=round(time.time() - t0, 1),
             output=out, comfyui=lora_path))
