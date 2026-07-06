"""Real LoRA training for Chroma — diffusers + peft, QLoRA nf4.

Chroma (lodestones/Chroma1-Base) = PRUNED and DE-DISTILLED Flux arch: same MMDiT
(double + single stream, latents packed 2×2 → 64 channels, Flux AE VAE 16 channels) BUT
**no CLIP** (text = T5-XXL only, with attention_mask) and **guidance-free** (the
vecteur de modulation vient d'un petit "approximator" interne, pas d'un guidance embed).

Reuses the local Flux components (ComfyUI): VAE `ae.safetensors` (Flux AE) + T5
`t5xxl_fp16.safetensors` — so only the Chroma DiT needs fetching. DiT config read
from the NON-gated repo `lodestones/Chroma1-Base` (subfolder transformer).

API verified (pipeline_chroma.py / transformer_chroma.py):
  transformer(hidden_states=packed[B,seq,64], timestep=sigma, encoder_hidden_states=
  T5[B,seq,4096], txt_ids=zeros[seq,3], img_ids[seq,3], attention_mask[B,seq]) ;
  flow-matching, timestep=sigma (×1000 internally), TARGET = x0 - x1.
"""
import gc
import hashlib
import os
import random
import time

from captioner import clean_path
from events import evt
from train_utils import make_lr_scheduler
# generic helpers reused from the Flux trainer (identical arch)
from flux_trainer import (
    _export_lora, _find_flux_components, _latent_image_ids, _load_square,
    _load_t5, _load_vae, _pack_latents, _sample_sigma,
)
from real_trainer import _list_dataset

_CHROMA_REPO = "lodestones/Chroma1-Base"  # config DiT (subfolder transformer), non gated
_T5_MAX = 512

_LORA_TARGETS = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
    "proj_out", "proj_mlp",
]


def _load_transformer(dit_path, precision, cache_dir, emit):
    """Chroma DiT in nf4 (or bf16), same on-disk cache as Flux."""
    import torch
    from diffusers import ChromaTransformer2DModel

    from quant import bnb_config, is_quantized, patch_single_file_fresh_quant

    device = "cuda"
    bnb = bnb_config(precision)
    if bnb is None:
        emit(evt("log", level="info", message="DiT Chroma bf16 (from_single_file)…"))
        tf = ChromaTransformer2DModel.from_single_file(
            dit_path, config=_CHROMA_REPO, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        return tf.to(device, dtype=torch.bfloat16)

    key = hashlib.sha1(f"{dit_path}|{os.path.getmtime(dit_path)}|{precision}".encode()).hexdigest()[:12]
    nf4_dir = os.path.join(cache_dir, f"chroma_{precision}_{key}")
    if os.path.isdir(nf4_dir):
        emit(evt("log", level="info", message=f"DiT Chroma {precision} cached → fast load…"))
        try:
            tf = ChromaTransformer2DModel.from_pretrained(nf4_dir, torch_dtype=torch.bfloat16)
            return tf.to(device)
        except Exception as e:
            emit(evt("log", level="warn", message=f"nf4 cache unreadable ({e}) → re-quantization"))

    emit(evt("log", level="info", message=f"Chroma DiT → {precision} (first time: read + quantization)…"))
    patch_single_file_fresh_quant()
    tf = ChromaTransformer2DModel.from_single_file(
        dit_path, config=_CHROMA_REPO, subfolder="transformer",
        quantization_config=bnb, torch_dtype=torch.bfloat16, device="cuda",
    )
    tf = tf.to(device)
    if is_quantized(precision):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tf.save_pretrained(nf4_dir)
            emit(evt("log", level="info", message="nf4 DiT cached (later runs are fast)"))
        except Exception as e:
            emit(evt("log", level="warn", message=f"nf4 cache not written: {e}"))
    return tf


def _encode_t5_masked(t5, t5_tok, prompt, device, dtype):
    """T5-XXL with attention_mask (Chroma masks the padding). Returns (emb[1,seq,4096],
    mask[1,seq])."""
    import torch

    toks = t5_tok(prompt, padding="max_length", max_length=_T5_MAX, truncation=True,
                  return_tensors="pt").to(device)
    with torch.no_grad():
        emb = t5(toks.input_ids, attention_mask=toks.attention_mask)[0]
    return emb.to("cpu", dtype), toks.attention_mask.to("cpu")


def run_chroma_training(cfg, emit, stop_event, family=None):
    import torch
    import torchvision.transforms as T
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    precision = getattr(cfg, "precision", "nf4") or "nf4"

    dit_path = clean_path(cfg.base_model)
    if not (dit_path.lower().endswith((".safetensors", ".ckpt")) and os.path.isfile(dit_path)):
        raise RuntimeError("Chroma: base_model must point to a local DiT (chroma*.safetensors).")

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"No images in {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) — Chroma QLoRA ({precision})"))

    res = int(cfg.resolution)
    if res % 16 != 0:
        res = (res // 16) * 16
    h = w = res // 8
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    cache_dir = os.path.join(cfg.output_dir, ".soma_cache")

    # Chroma reuses the Flux AE + T5 (we ignore the returned CLIP)
    vae_path, t5_path, _clip = _find_flux_components(dit_path, cfg)

    # ---------------- 1) cache latents (VAE Flux AE) ----------------
    emit(evt("log", level="info", message="Pre-computing latents (VAE)…"))
    vae = _load_vae(vae_path, torch.float32, emit).to(device)
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

    # ---------------- 2) cache embeddings texte (T5 + mask) ----------------
    emit(evt("log", level="info", message="Pre-computing text embeddings (T5)…"))
    t5, t5_tok = _load_t5(t5_path, dtype, emit)
    t5.to(device)
    default_cap = f"a photo of {cfg.instance_token} person"
    emb_cache = []
    for _, caption in data:
        emb, mask = _encode_t5_masked(t5, t5_tok, caption or default_cap, device, dtype)
        emb_cache.append((emb, mask))
    txt_ids = torch.zeros(_T5_MAX, 3, device=device, dtype=dtype)
    del t5
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
    sched = make_lr_scheduler(opt, cfg.max_steps, getattr(cfg, "lr_warmup_ratio", 0.05))
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
            emb, mask = emb_cache[i]
            emb = emb.to(device, dtype)
            mask = mask.to(device)
            x0 = torch.randn_like(x1)
            sigma = _sample_sigma()
            noisy = (1.0 - sigma) * x1 + sigma * x0
            tstep = torch.tensor([sigma], device=device, dtype=dtype)

            pred = transformer(
                hidden_states=noisy, timestep=tstep,
                encoder_hidden_states=emb, txt_ids=txt_ids, img_ids=img_ids,
                attention_mask=mask, return_dict=False,
            )[0]
            target = x0 - x1
            loss = torch.nn.functional.mse_loss(pred.float(), target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            sched.step()

            emit(evt("step", step=step, total_steps=cfg.max_steps, loss=round(loss.item(), 4),
                     lr=sched.get_last_lr()[0], secs=round(time.time() - t0, 1)))
        if stop_event.is_set():
            break

    out = os.path.join(cfg.output_dir, cfg.project_name)
    os.makedirs(out, exist_ok=True)
    transformer.save_pretrained(out)
    from families import get_family, soma_meta

    lora_path = _export_lora(transformer, out, cfg.project_name, emit,
                             meta=soma_meta(cfg, get_family(getattr(cfg, "arch", "chroma")), step))
    emit(evt("status", state="done", step=step, secs=round(time.time() - t0, 1),
             output=out, comfyui=lora_path))
