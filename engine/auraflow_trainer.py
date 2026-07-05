"""Vrai entraînement LoRA AuraFlow — diffusers + peft.

AuraFlow = MMDiT flow-matching, texte **UMT5** (pile-t5), VAE AutoencoderKL 4 canaux.
Distribué en repo diffusers → from_pretrained(subfolder). base_model défaut fal/AuraFlow-v0.3.

Vérifié (pipeline_aura_flow.py) : prompt_embeds = `text_encoder(ids, mask)[0] * mask`
(padding mis à zéro, PAS de mask passé au transformer) ; latents 4D non packés (in=4,
patch 2) ; flow STANDARD → timestep = sigma, CIBLE = x0 - x1 ; forward =
transformer(hidden_states[B,4,H,W], encoder_hidden_states, timestep).
"""
import gc
import os
import random
import time

from captioner import clean_path
from events import evt
from flux_trainer import _export_lora, _load_square, _sample_sigma
from real_trainer import _list_dataset

_AURA_REPO = "fal/AuraFlow-v0.3"
_TOK_MAX = 256
_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj"]


def _resolve(base_model):
    b = clean_path(base_model)
    return b if b else _AURA_REPO


def run_auraflow_training(cfg, emit, stop_event, family=None):
    import torch
    import torchvision.transforms as T
    from diffusers import AuraFlowTransformer2DModel, AutoencoderKL
    from peft import LoraConfig, get_peft_model
    from transformers import T5Tokenizer, UMT5EncoderModel

    from quant import bnb_config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    precision = getattr(cfg, "precision", "nf4") or "nf4"
    src = _resolve(cfg.base_model)

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"Aucune image dans {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) — AuraFlow QLoRA ({precision}) depuis {src}"))

    res = int(cfg.resolution)
    if res % 8 != 0:
        res = (res // 8) * 8
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    # ---------------- 1) cache latents (VAE KL 4ch) ----------------
    emit(evt("log", level="info", message="Pré-calcul des latents (VAE)…"))
    vae = AutoencoderKL.from_pretrained(src, subfolder="vae", torch_dtype=torch.float32).to(device)
    scaling = vae.config.scaling_factor
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    latents_cache = []
    with torch.no_grad():
        for path, caption in data:
            px = norm(_load_square(path, res)).unsqueeze(0).to(device, torch.float32)
            x1 = (vae.encode(px).latent_dist.sample() - shift) * scaling  # [1,4,h,w]
            latents_cache.append((x1.squeeze(0).to("cpu", dtype), caption))
    del vae
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 2) cache embeddings texte (UMT5, mask baked-in) ----------------
    emit(evt("log", level="info", message="Pré-calcul des embeddings texte (UMT5)…"))
    tok = T5Tokenizer.from_pretrained(src, subfolder="tokenizer")
    te = UMT5EncoderModel.from_pretrained(src, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
    default_cap = f"a photo of {cfg.instance_token} person"
    emb_cache = []
    with torch.no_grad():
        for _, caption in data:
            toks = tok(caption or default_cap, padding="max_length", max_length=_TOK_MAX,
                       truncation=True, return_tensors="pt").to(device)
            emb = te(**toks)[0]
            m = toks["attention_mask"].unsqueeze(-1).expand(emb.shape)
            emb = emb * m  # padding -> 0 (AuraFlow n'envoie pas de mask au transformer)
            emb_cache.append(emb.to("cpu", dtype))
    del te
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 3) transformer + QLoRA ----------------
    emit(evt("log", level="info", message=f"DiT AuraFlow ({precision})…"))
    bnb = bnb_config(precision)
    tkw = dict(subfolder="transformer", torch_dtype=dtype)
    if bnb is not None:
        tkw["quantization_config"] = bnb
    transformer = AuraFlowTransformer2DModel.from_pretrained(src, **tkw).to(device)
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
            x1 = latents_cache[i][0].unsqueeze(0).to(device, dtype)  # [1,4,h,w]
            emb = emb_cache[i].to(device, dtype)
            x0 = torch.randn_like(x1)
            sigma = _sample_sigma()
            noisy = (1.0 - sigma) * x1 + sigma * x0
            tstep = torch.tensor([sigma], device=device, dtype=dtype)

            pred = transformer(hidden_states=noisy, encoder_hidden_states=emb,
                               timestep=tstep, return_dict=False)[0]
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
                             meta=soma_meta(cfg, get_family(getattr(cfg, "arch", "auraflow")), step))
    emit(evt("status", state="done", step=step, secs=round(time.time() - t0, 1),
             output=out, comfyui=lora_path))
