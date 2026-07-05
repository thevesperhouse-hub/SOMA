"""Vrai entraînement LoRA PRX (Photoroom/prxpixel-t2i) — diffusers + peft.

PRX = DiT flow-matching compact (depth 16, hidden 1792, in_channels 16), texte encodé
par un **T5Gemma encoder** (context_in_dim 2304), VAE AutoencoderKL 16 canaux. Distribué
en **repo diffusers** (pas un single-file ComfyUI) → on charge les composants via
from_pretrained(subfolder). base_model = repo HF (défaut Photoroom/prxpixel-t2i) ou dossier
diffusers local.

API vérifiée (pipeline_prx.py / transformer_prx.py) : latents **4D non packés** [B,16,H,W]
(patchify interne, patch_size 2) ; texte = `text_encoder(ids, mask, output_hidden_states)
["last_hidden_state"]` + mask ; flow STANDARD (pas de négation) → timestep = **sigma**,
CIBLE = **x0 - x1** ; forward = transformer(hidden_states, timestep=sigma,
encoder_hidden_states, attention_mask).
"""
import gc
import os
import random
import time

from captioner import clean_path
from events import evt
from flux_trainer import _export_lora, _load_square, _sample_sigma
from real_trainer import _list_dataset

_PRX_REPO = "Photoroom/prxpixel-t2i"
_TOK_MAX = 256
# PRX = attention à projections FUSIONNÉES (img_qkv_proj / txt_kv_proj), pas to_q/k/v.
_LORA_TARGETS = ["img_qkv_proj", "txt_kv_proj", "to_out.0"]


def _resolve(base_model):
    """base_model = dossier diffusers local OU repo HF (défaut PRX)."""
    b = clean_path(base_model)
    return b if b else _PRX_REPO


def _load_transformer(src, precision, emit):
    import torch
    from diffusers import PRXTransformer2DModel

    from quant import bnb_config

    emit(evt("log", level="info", message=f"DiT PRX ({precision})…"))
    bnb = bnb_config(precision)
    kw = dict(subfolder="transformer", torch_dtype=torch.bfloat16)
    if bnb is not None:
        kw["quantization_config"] = bnb
    tf = PRXTransformer2DModel.from_pretrained(src, **kw)
    return tf.to("cuda") if bnb is None else tf.to("cuda")


def run_prx_training(cfg, emit, stop_event, family=None):
    import torch
    import torchvision.transforms as T
    from diffusers import AutoencoderKL
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    precision = getattr(cfg, "precision", "bf16") or "bf16"
    src = _resolve(cfg.base_model)

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"Aucune image dans {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) — PRX QLoRA ({precision}) depuis {src}"))

    res = int(cfg.resolution)
    if res % 16 != 0:
        res = (res // 16) * 16
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    # ---------------- 1) cache latents (VAE) ----------------
    emit(evt("log", level="info", message="Pré-calcul des latents (VAE)…"))
    vae = AutoencoderKL.from_pretrained(src, subfolder="vae", torch_dtype=torch.float32).to(device)
    scaling = vae.config.scaling_factor
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    latents_cache = []
    with torch.no_grad():
        for path, caption in data:
            px = norm(_load_square(path, res)).unsqueeze(0).to(device, torch.float32)
            raw = vae.encode(px).latent_dist.sample()
            x1 = (raw - shift) * scaling  # [1,16,h,w] non packé
            latents_cache.append((x1.squeeze(0).to("cpu", dtype), caption))
    del vae
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 2) cache embeddings texte (T5Gemma) ----------------
    emit(evt("log", level="info", message="Pré-calcul des embeddings texte (T5Gemma)…"))
    tok = AutoTokenizer.from_pretrained(src, subfolder="tokenizer")
    te = T5GemmaEncoder.from_pretrained(src, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
    default_cap = f"a photo of {cfg.instance_token} person"
    emb_cache = []
    with torch.no_grad():
        for _, caption in data:
            toks = tok(caption or default_cap, padding="max_length", max_length=_TOK_MAX,
                       truncation=True, return_tensors="pt").to(device)
            emb = te(input_ids=toks.input_ids, attention_mask=toks.attention_mask,
                     output_hidden_states=True)["last_hidden_state"]
            emb_cache.append((emb.to("cpu", dtype), toks.attention_mask.to("cpu")))
    del te
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 3) transformer + QLoRA ----------------
    transformer = _load_transformer(src, precision, emit)
    transformer.requires_grad_(False)
    lora = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, init_lora_weights="gaussian",
                      target_modules=_LORA_TARGETS)
    transformer = get_peft_model(transformer, lora)
    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    params = [p for p in transformer.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(params, lr=cfg.learning_rate)
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
            x1 = latents_cache[i][0].unsqueeze(0).to(device, dtype)  # [1,16,h,w]
            emb, mask = emb_cache[i]
            emb = emb.to(device, dtype)
            mask = mask.to(device)
            x0 = torch.randn_like(x1)
            sigma = _sample_sigma()
            noisy = (1.0 - sigma) * x1 + sigma * x0
            tstep = torch.tensor([sigma], device=device, dtype=dtype)

            pred = transformer(
                hidden_states=noisy, timestep=tstep,
                encoder_hidden_states=emb, attention_mask=mask,
                return_dict=False,
            )[0]
            target = x0 - x1  # flow standard : cible = noise - data
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
                             meta=soma_meta(cfg, get_family(getattr(cfg, "arch", "prx")), step))
    emit(evt("status", state="done", step=step, secs=round(time.time() - t0, 1),
             output=out, comfyui=lora_path))
