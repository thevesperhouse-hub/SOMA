"""Vrai entraînement LoRA Sana — diffusers + peft.

Sana = DiT flow-matching linéaire, texte **Gemma-2**, VAE **AutoencoderDC** (compression
32×, latent 32 canaux, déterministe). Distribué en repo diffusers → from_pretrained.
base_model = repo HF (défaut Efficient-Large-Model/Sana_1600M_1024px_diffusers) ou dossier local.

Vérifié (pipeline_sana.py / sana_transformer.py) : texte = `text_encoder(ids, mask)[0]`
(+mask) ; VAE DC-AE : x1 = encode(px) * scaling_factor (déterministe) ; flow STANDARD
(pas de négation) → timestep = sigma*1000*timestep_scale, CIBLE = x0 - x1 ; forward =
transformer(hidden_states, encoder_hidden_states, encoder_attention_mask, timestep).
"""
import gc
import os
import random
import time

from captioner import clean_path
from events import evt
from flux_trainer import _export_lora, _load_square, _sample_sigma
from real_trainer import _list_dataset

_SANA_REPO = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
_TOK_MAX = 300
_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


def _resolve(base_model):
    b = clean_path(base_model)
    return b if b else _SANA_REPO


def run_sana_training(cfg, emit, stop_event, family=None):
    import torch
    import torchvision.transforms as T
    from diffusers import AutoencoderDC, SanaTransformer2DModel
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    from quant import bnb_config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    precision = getattr(cfg, "precision", "bf16") or "bf16"
    src = _resolve(cfg.base_model)

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"No images in {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) — Sana QLoRA ({precision}) from {src}"))

    res = int(cfg.resolution)
    if res % 32 != 0:
        res = (res // 32) * 32
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    # ---------------- 1) cache latents (VAE DC-AE) ----------------
    emit(evt("log", level="info", message="Pre-computing latents (VAE DC-AE)…"))
    vae = AutoencoderDC.from_pretrained(src, subfolder="vae", torch_dtype=torch.float32).to(device)
    scaling = vae.config.scaling_factor
    latents_cache = []
    with torch.no_grad():
        for path, caption in data:
            px = norm(_load_square(path, res)).unsqueeze(0).to(device, torch.float32)
            enc = vae.encode(px)
            lat = enc.latent if hasattr(enc, "latent") else enc[0]
            x1 = lat * scaling  # [1,32,h,w] (déterministe)
            latents_cache.append((x1.squeeze(0).to("cpu", dtype), caption))
    del vae
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 2) cache embeddings texte (Gemma-2) ----------------
    emit(evt("log", level="info", message="Pre-computing text embeddings (Gemma-2)…"))
    tok = AutoTokenizer.from_pretrained(src, subfolder="tokenizer")
    te = AutoModel.from_pretrained(src, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
    default_cap = f"a photo of {cfg.instance_token} person"
    emb_cache = []
    with torch.no_grad():
        for _, caption in data:
            toks = tok(caption or default_cap, padding="max_length", max_length=_TOK_MAX,
                       truncation=True, return_tensors="pt").to(device)
            emb = te(toks.input_ids, attention_mask=toks.attention_mask)[0]  # last_hidden_state
            emb_cache.append((emb.to("cpu", dtype), toks.attention_mask.to("cpu")))
    del te
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 3) transformer + QLoRA ----------------
    emit(evt("log", level="info", message=f"DiT Sana ({precision})…"))
    bnb = bnb_config(precision)
    tkw = dict(subfolder="transformer", torch_dtype=dtype)
    if bnb is not None:
        tkw["quantization_config"] = bnb
        tkw["device_map"] = {"": 0}  # nf4 quant on GPU
    transformer = SanaTransformer2DModel.from_pretrained(src, **tkw)
    if bnb is None:
        transformer = transformer.to(device)
    ts_scale = float(getattr(transformer.config, "timestep_scale", 1.0) or 1.0)
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
            x1 = latents_cache[i][0].unsqueeze(0).to(device, dtype)
            emb, mask = emb_cache[i]
            emb = emb.to(device, dtype)
            mask = mask.to(device)
            x0 = torch.randn_like(x1)
            sigma = _sample_sigma()
            noisy = (1.0 - sigma) * x1 + sigma * x0
            tstep = torch.tensor([sigma * 1000.0 * ts_scale], device=device, dtype=dtype)

            pred = transformer(
                hidden_states=noisy, encoder_hidden_states=emb,
                encoder_attention_mask=mask, timestep=tstep, return_dict=False,
            )[0]
            if pred.shape[1] == 2 * x1.shape[1]:  # learned sigma éventuel
                pred = pred[:, : x1.shape[1]]
            target = x0 - x1  # standard flow: target = noise - data
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
                             meta=soma_meta(cfg, get_family(getattr(cfg, "arch", "sana")), step))
    emit(evt("status", state="done", step=step, secs=round(time.time() - t0, 1),
             output=out, comfyui=lora_path))
