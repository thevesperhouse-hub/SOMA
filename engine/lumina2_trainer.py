"""Real LoRA training for Lumina2 (Alpha-VLLM/Lumina-Image-2.0) — diffusers + peft.

Lumina2 = DiT flow-matching, texte encodé par **Gemma-2** (pas T5/CLIP), VAE = Flux AE
16 channels. Latents NON packés (le DiT patchifie en interne, patch_size 2, in_channels 16).

Convention flow (vérifiée dans pipeline_lumina2.py) — IDENTIQUE à Z-Image : la pipeline
NÈGE la sortie du transformer avant le scheduler (`noise_pred = -noise_pred`), et Lumina
utilise t=0=bruit / t=1=image → timestep modèle = **1 - sigma**, donc le transformer prédit
**(data - noise)** ⇒ CIBLE d'entraînement = **x1 - x0**. Embeds Gemma = `hidden_states[-2]`
avec system prompt + " <Prompt Start> " + caption (max 256 tokens) + attention_mask.
forward = transformer(hidden_states=latents[B,16,H,W], timestep=1-sigma,
encoder_hidden_states=gemma[B,seq,2304], encoder_attention_mask[B,seq]).

Config DiT lue depuis le repo (cap_feat_dim RÉEL = 2304, ≠ défaut 1024). Gemma + tokenizer
depuis le subfolder du repo (non gated). VAE = ae.safetensors local (Flux AE, réutilisé).
"""
import gc
import hashlib
import os
import random
import time

from captioner import clean_path
from events import evt
from flux_trainer import _export_lora, _load_square, _load_vae, _sample_sigma
from real_trainer import _list_dataset

_LUMINA_REPO = "Alpha-VLLM/Lumina-Image-2.0"
_SYSTEM_PROMPT = (
    "You are an assistant designed to generate superior images with the superior "
    "degree of image-text alignment based on textual prompts or user prompts."
)
_GEMMA_MAX = 256
_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


def _find_vae(dit_path, cfg):
    from zimage_trainer import _auto_component, _find_models_root

    root = _find_models_root(dit_path)
    if root is None:
        raise RuntimeError("Arbo ComfyUI models/ introuvable près du DiT Lumina2 (vae).")
    vae = clean_path(getattr(cfg, "zimage_vae", "")) or _auto_component(
        root, "vae", "ae.safetensors", ["ae", "flux"]
    )
    if not vae or not os.path.isfile(vae):
        raise RuntimeError(f"VAE (ae.safetensors, Flux AE) introuvable dans {root}")
    return vae


def _load_transformer(dit_path, precision, cache_dir, emit):
    import torch
    from diffusers import Lumina2Transformer2DModel

    from quant import bnb_config, is_quantized, patch_single_file_fresh_quant

    device = "cuda"
    bnb = bnb_config(precision)
    if bnb is None:
        emit(evt("log", level="info", message="DiT Lumina2 bf16 (from_single_file)…"))
        tf = Lumina2Transformer2DModel.from_single_file(
            dit_path, config=_LUMINA_REPO, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        return tf.to(device, dtype=torch.bfloat16)

    key = hashlib.sha1(f"{dit_path}|{os.path.getmtime(dit_path)}|{precision}".encode()).hexdigest()[:12]
    nf4_dir = os.path.join(cache_dir, f"lumina2_{precision}_{key}")
    if os.path.isdir(nf4_dir):
        emit(evt("log", level="info", message=f"DiT Lumina2 {precision} en cache → chargement rapide…"))
        try:
            tf = Lumina2Transformer2DModel.from_pretrained(nf4_dir, torch_dtype=torch.bfloat16)
            return tf.to(device)
        except Exception as e:
            emit(evt("log", level="warn", message=f"cache nf4 illisible ({e}) → re-quantization"))

    emit(evt("log", level="info", message=f"DiT Lumina2 → {precision} (1ère fois : lecture + quantization)…"))
    patch_single_file_fresh_quant()
    tf = Lumina2Transformer2DModel.from_single_file(
        dit_path, config=_LUMINA_REPO, subfolder="transformer",
        quantization_config=bnb, torch_dtype=torch.bfloat16, device="cuda",
    )
    tf = tf.to(device)
    if is_quantized(precision):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tf.save_pretrained(nf4_dir)
            emit(evt("log", level="info", message="DiT nf4 mis en cache (runs suivants rapides)"))
        except Exception as e:
            emit(evt("log", level="warn", message=f"cache nf4 non écrit: {e}"))
    return tf


def _load_gemma(precision, emit):
    import torch
    from transformers import AutoModel, AutoTokenizer

    from quant import bnb_config

    emit(evt("log", level="info", message="Text encoder Gemma-2 (Lumina2)…"))
    tok = AutoTokenizer.from_pretrained(_LUMINA_REPO, subfolder="tokenizer")
    tok.padding_side = "right"
    bnb = bnb_config(precision if precision != "bf16" else "nf4")
    kw = dict(torch_dtype=torch.bfloat16, subfolder="text_encoder")
    if bnb is not None:
        kw["quantization_config"] = bnb
        kw["device_map"] = {"": 0}
    te = AutoModel.from_pretrained(_LUMINA_REPO, **kw)
    if bnb is None:
        te = te.to("cuda")
    return te.eval(), tok


def _encode_gemma(te, tok, prompt, device, dtype):
    import torch

    txt = _SYSTEM_PROMPT + " <Prompt Start> " + prompt
    toks = tok(txt, padding="max_length", max_length=_GEMMA_MAX, truncation=True,
               return_tensors="pt").to(device)
    with torch.no_grad():
        out = te(toks.input_ids, attention_mask=toks.attention_mask, output_hidden_states=True)
    emb = out.hidden_states[-2]  # [1, seq, 2304]
    return emb.to("cpu", dtype), toks.attention_mask.to("cpu")


def run_lumina2_training(cfg, emit, stop_event, family=None):
    import torch
    import torchvision.transforms as T
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    precision = getattr(cfg, "precision", "nf4") or "nf4"

    dit_path = clean_path(cfg.base_model)
    if not (dit_path.lower().endswith((".safetensors", ".ckpt")) and os.path.isfile(dit_path)):
        raise RuntimeError("Lumina2 : base_model doit pointer un DiT local (lumina*.safetensors).")

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"No images in {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) — Lumina2 QLoRA ({precision})"))

    res = int(cfg.resolution)
    if res % 16 != 0:
        res = (res // 16) * 16
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    cache_dir = os.path.join(cfg.output_dir, ".soma_cache")
    vae_path = _find_vae(dit_path, cfg)

    # ---------------- 1) cache latents (Flux AE, 4D unpacked) ----------------
    emit(evt("log", level="info", message="Pre-computing latents (VAE)…"))
    vae = _load_vae(vae_path, torch.float32, emit).to(device)
    scaling = vae.config.scaling_factor
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    latents_cache = []
    with torch.no_grad():
        for path, caption in data:
            px = norm(_load_square(path, res)).unsqueeze(0).to(device, torch.float32)
            raw = vae.encode(px).latent_dist.sample()
            x1 = (raw - shift) * scaling  # [1,16,h,w] (NON packé)
            latents_cache.append((x1.squeeze(0).to("cpu", dtype), caption))
    del vae
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 2) cache embeddings texte (Gemma-2) ----------------
    emit(evt("log", level="info", message="Pre-computing text embeddings (Gemma-2)…"))
    te, tok = _load_gemma(precision, emit)
    default_cap = f"a portrait photo of {cfg.instance_token} person"
    emb_cache = []
    for _, caption in data:
        emb, mask = _encode_gemma(te, tok, caption or default_cap, device, dtype)
        emb_cache.append((emb, mask))
    del te
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 3) transformer nf4 + QLoRA ----------------
    transformer = _load_transformer(dit_path, precision, cache_dir, emit)
    transformer.requires_grad_(False)
    lora = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, init_lora_weights="gaussian",
                      target_modules=_LORA_TARGETS)
    transformer = get_peft_model(transformer, lora)
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
            tstep = torch.tensor([1.0 - sigma], device=device, dtype=dtype)  # Lumina : t=1-sigma

            pred = transformer(
                hidden_states=noisy, timestep=tstep,
                encoder_hidden_states=emb, encoder_attention_mask=mask,
                return_dict=False,
            )[0]
            target = x1 - x0  # Lumina/Z-Image : target = data - noise
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
                             meta=soma_meta(cfg, get_family(getattr(cfg, "arch", "lumina2")), step))
    emit(evt("status", state="done", step=step, secs=round(time.time() - t0, 1),
             output=out, comfyui=lora_path))
