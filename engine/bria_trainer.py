"""Vrai entraînement LoRA Bria (BRIA-3.x) — diffusers + peft, QLoRA nf4.

Bria = archi Flux (double+single stream, latents packés 2×2 → 64 canaux, VAE KL 16ch)
mais **T5-only** (pas de CLIP/pooled) et **sans guidance** (pas distillé). Distribué en
repo diffusers → from_pretrained(subfolder). base_model défaut briaai/BRIA-3.2 (gated).

Vérifié (pipeline_bria.py / transformer_bria.py) : forward = transformer(hidden_states
[B,seq,64], timestep, encoder_hidden_states=T5[B,seq,4096], txt_ids[seq,3], img_ids[seq,3])
— PAS de pooled/guidance/mask. Flow standard : timestep = sigma*1000, CIBLE = x0 - x1.
"""
import gc
import os
import random
import time

from captioner import clean_path
from events import evt
from flux_trainer import _export_lora, _latent_image_ids, _load_square, _pack_latents, _sample_sigma
from real_trainer import _list_dataset

_BRIA_REPO = "briaai/BRIA-3.2"
_T5_MAX = 128
_LORA_TARGETS = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
    "proj_out", "proj_mlp",
]


def _resolve(base_model):
    b = clean_path(base_model)
    return b if b else _BRIA_REPO


def run_bria_training(cfg, emit, stop_event, family=None):
    import torch
    import torchvision.transforms as T
    from diffusers import AutoencoderKL, BriaTransformer2DModel
    from peft import LoraConfig, get_peft_model
    from transformers import T5EncoderModel, T5TokenizerFast

    from quant import bnb_config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    precision = getattr(cfg, "precision", "nf4") or "nf4"
    src = _resolve(cfg.base_model)

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"No images in {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) — Bria QLoRA ({precision}) from {src}"))

    res = int(cfg.resolution)
    if res % 16 != 0:
        res = (res // 16) * 16
    h = w = res // 8
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    # ---------------- 1) cache latents (VAE KL 16ch) ----------------
    emit(evt("log", level="info", message="Pre-computing latents (VAE)…"))
    vae = AutoencoderKL.from_pretrained(src, subfolder="vae", torch_dtype=torch.float32).to(device)
    scaling = vae.config.scaling_factor
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    latents_cache = []
    with torch.no_grad():
        for path, caption in data:
            px = norm(_load_square(path, res)).unsqueeze(0).to(device, torch.float32)
            raw = vae.encode(px).latent_dist.sample()
            x1 = (raw - shift) * scaling
            packed = _pack_latents(x1, 1, 16, h, w).squeeze(0)  # [seq,64]
            latents_cache.append((packed.to("cpu", dtype), caption))
    img_ids = _latent_image_ids(h // 2, w // 2, device, dtype)
    del vae
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 2) cache embeddings texte (T5) ----------------
    emit(evt("log", level="info", message="Pre-computing text embeddings (T5)…"))
    tok = T5TokenizerFast.from_pretrained(src, subfolder="tokenizer")
    te = T5EncoderModel.from_pretrained(src, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
    default_cap = f"a photo of {cfg.instance_token} person"
    emb_cache = []
    with torch.no_grad():
        for _, caption in data:
            ids = tok(caption or default_cap, padding="max_length", max_length=_T5_MAX,
                      truncation=True, return_tensors="pt").input_ids.to(device)
            emb = te(ids)[0]
            emb_cache.append(emb.to("cpu", dtype))
    txt_ids = torch.zeros(_T5_MAX, 3, device=device, dtype=dtype)
    del te
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 3) transformer nf4 + QLoRA ----------------
    emit(evt("log", level="info", message=f"DiT Bria ({precision})…"))
    bnb = bnb_config(precision)
    tkw = dict(subfolder="transformer", torch_dtype=dtype)
    if bnb is not None:
        tkw["quantization_config"] = bnb
        tkw["device_map"] = {"": 0}  # nf4 quant on GPU
    transformer = BriaTransformer2DModel.from_pretrained(src, **tkw)
    if bnb is None:
        transformer = transformer.to(device)
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
            x1 = latents_cache[i][0].unsqueeze(0).to(device, dtype)  # [1,seq,64]
            emb = emb_cache[i].to(device, dtype)
            x0 = torch.randn_like(x1)
            sigma = _sample_sigma()
            noisy = (1.0 - sigma) * x1 + sigma * x0
            tstep = torch.tensor([sigma * 1000.0], device=device, dtype=dtype)

            pred = transformer(
                hidden_states=noisy, timestep=tstep, encoder_hidden_states=emb,
                txt_ids=txt_ids, img_ids=img_ids, return_dict=False,
            )[0]
            target = x0 - x1
            loss = torch.nn.functional.mse_loss(pred.float(), target.float())
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
                             meta=soma_meta(cfg, get_family(getattr(cfg, "arch", "bria")), step))
    emit(evt("status", state="done", step=step, secs=round(time.time() - t0, 1),
             output=out, comfyui=lora_path))
