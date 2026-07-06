"""Real LoRA training for Flux.1-dev — diffusers + peft, QLoRA nf4 (PAS Ostris).

Flux = MMDiT 12B, objectif flow-matching. Sur 16 Go : le transformer bf16 fait
23,8 Go -> on le QUANTIFIE en nf4 (~6,8 Go) + gradient checkpointing + on cache
latents & text embeddings then unload VAE/T5/CLIP (only the nf4 DiT remains).

LOCAL files (ComfyUI, zero heavy download): DiT (flux1-dev.safetensors), VAE
(ae.safetensors = Flux AE), T5 (t5xxl_fp16.safetensors), CLIP-L (clip_l.safetensors).
Configs: bundled DiT (model_configs/flux_transformer), VAE via the Z-Image repo (same
Flux AE, non gated), CLIP `openai/clip-vit-large-patch14`, T5 `google/t5-v1_1-xxl`
(public, a few MB). API verified by reading pipeline_flux.py + smoke test.

Objective (rectified flow): x1 = (vae.encode-shift)*scaling PACKED 2×2 (64 channels);
x0 ~ N(0,1); logit-normal sigma; noisy = (1-sigma)*x1 + sigma*x0; model timestep
= sigma; fixed guidance (dev = guidance_embeds); TARGET = x0 - x1 (Flux does not negate).
"""
import gc
import hashlib
import io
import os
import random
import time

from captioner import clean_path
from events import evt
from real_trainer import _buckets_for_resolution, _list_dataset  # noqa: F401 (generic helpers)

_CFG_DIR = os.path.join(os.path.dirname(__file__), "model_configs", "flux_transformer")
_VAE_CONFIG_REPO = "Tongyi-MAI/Z-Image-Turbo"  # same Flux AE 16 channels, NON-gated repo
_CLIP_REPO = "openai/clip-vit-large-patch14"
_T5_REPO = "google/t5-v1_1-xxl"
_GUIDANCE = 1.0  # Flux.1-dev is guidance-distilled -> fixed value during training
_T5_MAX = 512

_LORA_TARGETS = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
    "proj_out", "proj_mlp",
]


# ------------------------------------------------------------------ packing Flux
def _pack_latents(latents, b, c, h, w):
    import torch  # noqa: F401

    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)


def _latent_image_ids(h2, w2, device, dtype):
    import torch

    ids = torch.zeros(h2, w2, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(h2)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w2)[None, :]
    return ids.reshape(h2 * w2, 3).to(device=device, dtype=dtype)


def _load_square(path, res):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((res, res), Image.LANCZOS)


# ------------------------------------------------------------------ components
def _load_transformer(dit_path, precision, cache_dir, emit):
    """Flux DiT in nf4 (or bf16). On-disk cache of the quantized model: quantized once
    (reading 24 GB, ~4 min) then re-read ~7 GB instantly on later runs."""
    import torch
    from diffusers import FluxTransformer2DModel

    from quant import bnb_config, is_quantized, patch_single_file_fresh_quant

    device = "cuda"
    bnb = bnb_config(precision)
    if bnb is None:  # bf16 : ne tient PAS sur 16 Go mais on laisse le choix (grosses cartes)
        emit(evt("log", level="info", message="DiT Flux bf16 (from_single_file)…"))
        tf = FluxTransformer2DModel.from_single_file(dit_path, config=_CFG_DIR, torch_dtype=torch.bfloat16)
        return tf.to(device, dtype=torch.bfloat16)

    key = hashlib.sha1(f"{dit_path}|{os.path.getmtime(dit_path)}|{precision}".encode()).hexdigest()[:12]
    nf4_dir = os.path.join(cache_dir, f"flux_{precision}_{key}")
    if os.path.isdir(nf4_dir):
        emit(evt("log", level="info", message=f"DiT Flux {precision} cached → fast load…"))
        try:
            tf = FluxTransformer2DModel.from_pretrained(nf4_dir, torch_dtype=torch.bfloat16)
            return tf.to(device)
        except Exception as e:
            emit(evt("log", level="warn", message=f"nf4 cache unreadable ({e}) → re-quantization"))

    emit(evt("log", level="info", message=f"Flux DiT → {precision} (reading 24 GB, ~4 min the first time)…"))
    patch_single_file_fresh_quant()
    # device="cuda" -> la quantization nf4 se fait sur GPU (sinon CPU = interminable)
    tf = FluxTransformer2DModel.from_single_file(
        dit_path, config=_CFG_DIR, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device="cuda",
    )
    tf = tf.to(device)
    if is_quantized(precision):  # save for the next runs
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tf.save_pretrained(nf4_dir)
            emit(evt("log", level="info", message="nf4 DiT cached (later runs are fast)"))
        except Exception as e:
            emit(evt("log", level="warn", message=f"nf4 cache not written: {e}"))
    return tf


def _load_vae(vae_path, dtype, emit):
    from diffusers import AutoencoderKL

    emit(evt("log", level="info", message=f"VAE Flux: {os.path.basename(vae_path)}"))
    return AutoencoderKL.from_single_file(
        vae_path, config=_VAE_CONFIG_REPO, subfolder="vae", torch_dtype=dtype
    )


def _load_clip(clip_path, dtype, emit):
    import safetensors.torch as st
    from transformers import CLIPTextModel, CLIPTokenizer

    from accelerate import init_empty_weights

    emit(evt("log", level="info", message=f"CLIP-L: {os.path.basename(clip_path)}"))
    from transformers import CLIPTextConfig

    cfg = CLIPTextConfig.from_pretrained(_CLIP_REPO)
    with init_empty_weights(include_buffers=False):
        clip = CLIPTextModel(cfg)
    sd = st.load_file(clip_path)
    clip.load_state_dict(sd, strict=False, assign=True)
    tok = CLIPTokenizer.from_pretrained(_CLIP_REPO)
    return clip.to(dtype), tok


def _load_t5(t5_path, dtype, emit):
    import safetensors.torch as st
    from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

    from accelerate import init_empty_weights

    emit(evt("log", level="info", message=f"T5-XXL: {os.path.basename(t5_path)}"))
    cfg = AutoConfig.from_pretrained(_T5_REPO)
    with init_empty_weights(include_buffers=False):
        t5 = T5EncoderModel(cfg)
    sd = st.load_file(t5_path)
    t5.load_state_dict(sd, strict=False, assign=True)
    tok = AutoTokenizer.from_pretrained(_T5_REPO)
    return t5.to(dtype), tok


def _find_flux_components(dit_path, cfg):
    """Locate VAE + T5 + CLIP in the ComfyUI tree (relative to the DiT)."""
    from zimage_trainer import _auto_component, _find_models_root

    root = _find_models_root(dit_path)
    if root is None:
        raise RuntimeError("ComfyUI models/ tree not found near the Flux DiT (vae, text_encoders).")
    vae = clean_path(getattr(cfg, "zimage_vae", "")) or _auto_component(root, "vae", "ae.safetensors", ["ae"])
    t5 = _auto_component(root, "text_encoders", "t5xxl_fp16.safetensors", ["t5xxl", "t5"])
    clip = _auto_component(root, "text_encoders", "clip_l.safetensors", ["clip_l", "clip-l"])
    for name, p in (("VAE", vae), ("T5", t5), ("CLIP-L", clip)):
        if not p or not os.path.isfile(p):
            raise RuntimeError(f"{name} Flux not found in {root}")
    return vae, t5, clip


def _sample_sigma():
    import torch

    return torch.sigmoid(torch.randn(1)).item()


def _export_lora(transformer, out_dir, name, emit, meta=None):
    try:
        import torch
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        sd = get_peft_model_state_dict(transformer)
        out = {}
        for k, v in sd.items():
            k = k.replace("base_model.model.", "")
            if not k.startswith("transformer."):
                k = "transformer." + k
            out[k] = v.detach().to("cpu", torch.float16)
        path = os.path.join(out_dir, name + ".safetensors")
        save_file(out, path, metadata=meta or None)
        emit(evt("log", level="info", message=f"Flux LoRA exported: {name}.safetensors"))
        return path
    except Exception as e:
        emit(evt("log", level="warn", message=f"LoRA export failed: {e}"))
        return None


def run_flux_training(cfg, emit, stop_event, family=None):
    import torch
    import torchvision.transforms as T
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    precision = getattr(cfg, "precision", "nf4") or "nf4"

    dit_path = clean_path(cfg.base_model)
    if not (dit_path.lower().endswith((".safetensors", ".ckpt")) and os.path.isfile(dit_path)):
        raise RuntimeError("Flux: base_model must point to a local DiT (flux1-dev.safetensors).")

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"No images in {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) — Flux QLoRA ({precision})"))

    res = int(cfg.resolution)
    if res % 16 != 0:
        res = (res // 16) * 16
    h = w = res // 8  # dims latentes VAE
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    cache_dir = os.path.join(cfg.output_dir, ".soma_cache")

    vae_path, t5_path, clip_path = _find_flux_components(dit_path, cfg)

    # ---------------- 1) cache latents (VAE) ----------------
    emit(evt("log", level="info", message="Pre-computing latents (VAE)…"))
    vae = _load_vae(vae_path, torch.float32, emit).to(device)
    scaling = vae.config.scaling_factor
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    latents_cache = []
    with torch.no_grad():
        for path, caption in data:
            px = norm(_load_square(path, res)).unsqueeze(0).to(device, torch.float32)
            raw = vae.encode(px).latent_dist.sample()
            x1 = (raw - shift) * scaling  # [1,16,h,w]
            packed = _pack_latents(x1, 1, 16, h, w).squeeze(0)  # [seq,64]
            latents_cache.append((packed.to("cpu", dtype), caption))
    img_ids = _latent_image_ids(h // 2, w // 2, device, dtype)  # constant (fixed resolution)
    del vae
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 2) cache embeddings texte (CLIP + T5) ----------------
    emit(evt("log", level="info", message="Pre-computing text embeddings (CLIP + T5)…"))
    clip, clip_tok = _load_clip(clip_path, dtype, emit)
    t5, t5_tok = _load_t5(t5_path, dtype, emit)
    clip.to(device); t5.to(device)
    default_cap = f"a photo of {cfg.instance_token} person"
    emb_cache = []
    with torch.no_grad():
        for _, caption in data:
            cap = caption or default_cap
            cid = clip_tok(cap, padding="max_length", max_length=77, truncation=True,
                           return_tensors="pt").input_ids.to(device)
            pooled = clip(cid, output_hidden_states=False).pooler_output.to("cpu", dtype)  # [1,768]
            tid = t5_tok(cap, padding="max_length", max_length=_T5_MAX, truncation=True,
                         return_tensors="pt").input_ids.to(device)
            t5emb = t5(tid)[0].to("cpu", dtype)  # [1,seq,4096]
            emb_cache.append((pooled, t5emb))
    txt_ids = torch.zeros(_T5_MAX, 3, device=device, dtype=dtype)
    del clip, t5
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 3) transformer nf4 + QLoRA ----------------
    transformer = _load_transformer(dit_path, precision, cache_dir, emit)
    transformer.requires_grad_(False)
    lora = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, init_lora_weights="gaussian",
                      target_modules=_LORA_TARGETS)
    transformer = get_peft_model(transformer, lora)
    transformer.enable_gradient_checkpointing()  # OBLIGATOIRE sur 16 Go
    params = [p for p in transformer.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(params, lr=cfg.learning_rate)
        emit(evt("log", level="info", message="Optimizer: AdamW8bit"))
    except Exception:
        opt = torch.optim.AdamW(params, lr=cfg.learning_rate)

    guidance = torch.full([1], _GUIDANCE, device=device, dtype=torch.float32)
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
            pooled, t5emb = emb_cache[i]
            pooled = pooled.to(device, dtype)
            t5emb = t5emb.to(device, dtype)
            x0 = torch.randn_like(x1)
            sigma = _sample_sigma()
            noisy = (1.0 - sigma) * x1 + sigma * x0
            tstep = torch.tensor([sigma], device=device, dtype=dtype)

            pred = transformer(
                hidden_states=noisy, timestep=tstep, guidance=guidance,
                pooled_projections=pooled, encoder_hidden_states=t5emb,
                txt_ids=txt_ids, img_ids=img_ids, return_dict=False,
            )[0]
            target = x0 - x1  # Flux : cible = noise - data
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
    transformer.save_pretrained(out)  # adaptateur PEFT
    from families import get_family, soma_meta

    lora_path = _export_lora(transformer, out, cfg.project_name, emit,
                             meta=soma_meta(cfg, get_family(getattr(cfg, "arch", "flux")), step))
    emit(evt("status", state="done", step=step, secs=round(time.time() - t0, 1),
             output=out, comfyui=lora_path))
