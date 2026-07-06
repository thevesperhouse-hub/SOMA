"""Real LoRA training for CogView4 — diffusers + peft, QLoRA nf4.

CogView4 = DiT flow-matching, texte **GLM-4-9B** (`hidden_states[-2]`), VAE AutoencoderKL
16 channels, + micro-conditionnement SDXL (original_size / target_size / crop_coords).
Encoder 9B → chargé en nf4 pour tenir la VRAM pendant le pré-calcul, puis déchargé.
Repo diffusers → from_pretrained(subfolder). base_model default THUDM/CogView4-6B.

Verified (pipeline_cogview4.py) : latents 4D (in=16, patch 2) ; flow STANDARD → cible
x0 - x1, timestep = sigma*1000 ; forward = transformer(hidden_states, encoder_hidden_states,
timestep, original_size, target_size, crop_coords).
"""
import gc
import os
import random
import time

from captioner import clean_path
from events import evt
from flux_trainer import _export_lora, _load_square, _sample_sigma
from real_trainer import _list_dataset

_COGVIEW_REPO = "THUDM/CogView4-6B"
_TOK_MAX = 256
_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


def _resolve(base_model):
    b = clean_path(base_model)
    return b if b else _COGVIEW_REPO


def run_cogview4_training(cfg, emit, stop_event, family=None):
    import torch
    import torchvision.transforms as T
    from diffusers import AutoencoderKL, CogView4Transformer2DModel
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    from quant import bnb_config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    precision = getattr(cfg, "precision", "nf4") or "nf4"
    src = _resolve(cfg.base_model)

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"No images in {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) — CogView4 QLoRA ({precision}) from {src}"))

    res = int(cfg.resolution)
    if res % 32 != 0:
        res = (res // 32) * 32
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    # ---------------- 1) cache latents (VAE KL) ----------------
    emit(evt("log", level="info", message="Pre-computing latents (VAE)…"))
    vae = AutoencoderKL.from_pretrained(src, subfolder="vae", torch_dtype=torch.float32).to(device)
    scaling = vae.config.scaling_factor
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    latents_cache = []
    with torch.no_grad():
        for path, caption in data:
            px = norm(_load_square(path, res)).unsqueeze(0).to(device, torch.float32)
            x1 = (vae.encode(px).latent_dist.sample() - shift) * scaling  # [1,16,h,w]
            latents_cache.append((x1.squeeze(0).to("cpu", dtype), caption))
    del vae
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 2) cache embeddings texte (GLM-4-9B nf4) ----------------
    emit(evt("log", level="info", message="Pre-computing text embeddings (GLM-4-9B)…"))
    tok = AutoTokenizer.from_pretrained(src, subfolder="tokenizer")
    bnb_te = bnb_config(precision if precision != "bf16" else "nf4")
    tekw = dict(subfolder="text_encoder", torch_dtype=dtype)
    if bnb_te is not None:
        tekw["quantization_config"] = bnb_te
        tekw["device_map"] = {"": 0}
    te = AutoModel.from_pretrained(src, **tekw)
    if bnb_te is None:
        te = te.to(device)
    te.eval()
    default_cap = f"a photo of {cfg.instance_token} person"
    emb_cache = []
    with torch.no_grad():
        for _, caption in data:
            ids = tok(caption or default_cap, padding="max_length", max_length=_TOK_MAX,
                      truncation=True, return_tensors="pt").input_ids.to(device)
            emb = te(ids, output_hidden_states=True).hidden_states[-2]
            emb_cache.append(emb.to("cpu", dtype))
    del te
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 3) transformer nf4 + QLoRA ----------------
    emit(evt("log", level="info", message=f"DiT CogView4 ({precision})…"))
    bnb = bnb_config(precision)
    tkw = dict(subfolder="transformer", torch_dtype=dtype)
    if bnb is not None:
        tkw["quantization_config"] = bnb
        tkw["device_map"] = {"": 0}  # nf4 quant on GPU
    transformer = CogView4Transformer2DModel.from_pretrained(src, **tkw)
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

    osize = torch.tensor([[res, res]], device=device, dtype=dtype)  # micro-cond fixe
    crop = torch.tensor([[0, 0]], device=device, dtype=dtype)
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
            emb = emb_cache[i].to(device, dtype)
            x0 = torch.randn_like(x1)
            sigma = _sample_sigma()
            noisy = (1.0 - sigma) * x1 + sigma * x0
            tstep = torch.tensor([sigma * 1000.0], device=device, dtype=dtype)

            pred = transformer(
                hidden_states=noisy, encoder_hidden_states=emb, timestep=tstep,
                original_size=osize, target_size=osize, crop_coords=crop,
                return_dict=False,
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
                             meta=soma_meta(cfg, get_family(getattr(cfg, "arch", "cogview4")), step))
    emit(evt("status", state="done", step=step, secs=round(time.time() - t0, 1),
             output=out, comfyui=lora_path))
