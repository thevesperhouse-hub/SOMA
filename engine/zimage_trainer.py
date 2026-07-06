"""Real LoRA training for Z-Image Turbo — diffusers ZImagePipeline + peft (PAS Ostris).

Z-Image = DiT single-stream 6B, objectif flow-matching (rectified flow), pas epsilon.
Composants : VAE (AutoencoderKL, latents 16 channels avec shift_factor), text encoder
Qwen (chat template, on prend hidden_states[-2], dim 2560), transformer 6B.

⚠️ 16 GB constraint: can't hold VAE + Qwen + 6B transformer AT THE SAME TIME
with gradients. Strategy = we PRE-COMPUTE once the VAE latents and the
text embeddings (+ the preview prompt), unload the text encoder and keep only
the transformer on the GPU for the training loop. The VAE stays warm
(CPU) and moves back to the GPU just for a preview.

Objective details (verified against the diffusers pipeline):
- "model" latent x1 = (vae.encode(img).sample() - shift_factor) * scaling_factor
- bruit x0 ~ N(0,1) ; sigma ∈ (0,1) (logit-normal)
- noisy latent = sigma*x0 + (1-sigma)*x1   (sigma=1 -> pure noise, sigma=0 -> data)
- timestep passed to the model = 1 - sigma  (the pipeline uses (1000 - t)/1000)
- the pipeline NEGATES the transformer output before the scheduler => the model predicts
  natively (data - noise). So the training target = x1 - x0.
"""
import base64
import gc
import io
import math
import os
import random
import time

from captioner import clean_path
from events import evt

# Generic helpers reused from the SDXL trainer (dataset + ratio bucketing).
# Les buckets sont des multiples de 64 -> compatibles Z-Image (exige multiples de 16).
from real_trainer import _buckets_for_resolution, _list_dataset, _load_bucketed

ZIMAGE_DEFAULT = "Tongyi-MAI/Z-Image-Turbo"
QWEN_TE_REPO = "Qwen/Qwen3-4B"  # config + tokenizer seulement (quelques Mo), PAS les poids

# LoRA sur les projections d'attention + FFN des blocs du transformer (pas les
# embedders/adaLN/final_layer, qui ne matchent pas ces suffixes).
_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]


def _find_models_root(dit_path):
    """Walk up the tree from the DiT to a ComfyUI `models` folder
    (contenant vae/ et text_encoders/)."""
    d = os.path.dirname(os.path.abspath(dit_path))
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "vae")) and os.path.isdir(os.path.join(d, "text_encoders")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _auto_component(root, subdir, preferred, patterns):
    """Find a component (vae/text_encoder) in models/<subdir>: preferred name
    first, otherwise the 1st .safetensors matching a pattern (avoids gguf/fp8)."""
    import glob as _glob

    folder = os.path.join(root, subdir)
    pref = os.path.join(folder, preferred)
    if os.path.isfile(pref):
        return pref
    cands = sorted(_glob.glob(os.path.join(folder, "*.safetensors")))
    for pat in patterns:
        for c in cands:
            name = os.path.basename(c).lower()
            if pat in name and "fp8" not in name and "scaled" not in name:
                return c
    return None


def _build_pipeline_from_files(dit_path, vae_path, te_path, dtype, emit, need_te=True, precision="bf16"):
    """Assemble ZImagePipeline from 3 separate ComfyUI files: Z-Image DiT,
    Flux VAE (ae.zimage), Qwen3-4B text encoder. Downloads ONLY small
    configs/tokenizer (pas les poids lourds, on a tout en local).
    need_te=False (embedding cache present) -> we DO NOT LOAD Qwen (8 GB).
    precision int8/nf4 -> quantifie le DiT (moins de VRAM, sur petites cartes)."""
    import safetensors.torch as st
    import torch
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        ZImagePipeline,
        ZImageTransformer2DModel,
    )
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from quant import bnb_config, patch_single_file_fresh_quant

    emit(evt("log", level="info", message=f"DiT (single-file): {os.path.basename(dit_path)}"))
    bnb = bnb_config(precision)
    if bnb is not None:
        patch_single_file_fresh_quant()
        emit(evt("log", level="info", message=f"Quantization DiT: {precision}"))
        transformer = ZImageTransformer2DModel.from_single_file(
            dit_path, quantization_config=bnb, torch_dtype=dtype
        )
    else:
        transformer = ZImageTransformer2DModel.from_single_file(dit_path, torch_dtype=dtype)

    emit(evt("log", level="info", message=f"VAE (Flux AE): {os.path.basename(vae_path)}"))
    # ⚠️ without an explicit config, from_single_file builds an SD VAE 4 channels ->
    # mismatch (le VAE Z-Image/Flux fait 16 channels). On force la config du repo.
    vae = AutoencoderKL.from_single_file(
        vae_path, config=ZIMAGE_DEFAULT, subfolder="vae", torch_dtype=dtype
    )

    if need_te:
        emit(evt("log", level="info", message=f"Text encoder Qwen3: {os.path.basename(te_path)}"))
        from accelerate import init_empty_weights

        te_cfg = AutoConfig.from_pretrained(QWEN_TE_REPO)
        # init on meta device (instant, no 16 GB fp32 alloc, no random init);
        # include_buffers=False -> the buffers (rotary inv_freq) stay real.
        with init_empty_weights(include_buffers=False):
            te = AutoModel.from_config(te_cfg)  # Qwen3Model
        sd = st.load_file(te_path)
        sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
        sd.pop("lm_head.weight", None)
        # assign=True : place directement les tenseurs bf16 du disque (pas de copie fp32)
        missing, unexpected = te.load_state_dict(sd, strict=False, assign=True)
        real_missing = [k for k in missing if "rotary" not in k and "inv_freq" not in k]
        if real_missing or unexpected:
            emit(evt("log", level="warn",
                     message=f"Qwen load: {len(real_missing)} manquants, {len(unexpected)} inattendus"))
        tokenizer = AutoTokenizer.from_pretrained(QWEN_TE_REPO)
    else:
        emit(evt("log", level="info", message="Embedding cache present → Qwen not loaded (time saved)"))
        te = None
        tokenizer = None

    scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
    )
    return ZImagePipeline(
        scheduler=scheduler, vae=vae, text_encoder=te, tokenizer=tokenizer, transformer=transformer
    )


def _load_pipeline(cfg, dtype, emit, need_te=True):
    """Load the Z-Image pipeline. Priority: local ComfyUI files (0 download
    lourd) si base_model pointe un DiT .safetensors ; sinon repo diffusers HF.
    need_te=False -> skip loading the Qwen text encoder (cache present)."""
    base = clean_path(cfg.base_model) or ZIMAGE_DEFAULT
    is_local = base.lower().endswith((".safetensors", ".ckpt")) and os.path.isfile(base)
    if not is_local:
        from diffusers import ZImagePipeline

        emit(evt("log", level="info", message=f"Z-Image : from_pretrained {base} (repo HF)…"))
        pipe = ZImagePipeline.from_pretrained(base, torch_dtype=dtype)
        if not need_te and getattr(pipe, "text_encoder", None) is not None:
            del pipe.text_encoder
            pipe.text_encoder = None
        return pipe

    # Separate files: locate VAE + text encoder
    vae_path = clean_path(cfg.zimage_vae)
    te_path = clean_path(cfg.zimage_text_encoder)
    if not vae_path or not te_path:
        root = _find_models_root(base)
        if root is None:
            raise RuntimeError(
                "Local DiT detected but couldn't find the ComfyUI models/ tree "
                "(vae, text_encoders). Renseigne zimage_vae et zimage_text_encoder."
            )
        vae_path = vae_path or _auto_component(root, "vae", "ae.zimage.safetensors", ["zimage", "ae"])
        te_path = te_path or _auto_component(root, "text_encoders", "qwen_3_4b.safetensors", ["qwen_3", "qwen3"])
    if not vae_path or not os.path.isfile(vae_path):
        raise RuntimeError(f"Z-Image VAE not found (looked for ae.zimage). Given: {vae_path!r}")
    if need_te and (not te_path or not os.path.isfile(te_path)):
        raise RuntimeError(f"Qwen3 text encoder not found. Given: {te_path!r}")
    return _build_pipeline_from_files(
        base, vae_path, te_path, dtype, emit, need_te=need_te,
        precision=getattr(cfg, "precision", "bf16"),
    )


def _cache_key(data, cfg):
    """Fingerprint of the dataset (paths + mtime + size + caption) + resolution +
    token + preview prompt. Any image or .txt change invalidates the cache."""
    import hashlib

    h = hashlib.sha1()
    h.update(f"{cfg.resolution}|{cfg.instance_token}|{cfg.sample_prompt}".encode("utf-8", "ignore"))
    for path, cap in data:
        try:
            st_ = os.stat(path)
            sig = f"{path}|{int(st_.st_mtime)}|{st_.st_size}|{cap}"
        except OSError:
            sig = f"{path}|0|0|{cap}"
        h.update(sig.encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def _load_cache(cache_file, emit):
    import torch

    try:
        blob = torch.load(cache_file, map_location="cpu")
        return blob["latents"], blob["embeds"], blob["sample"]
    except Exception as e:
        emit(evt("log", level="warn", message=f"cache illisible ({e}) → recalcul"))
        return None


def _save_cache(cache_file, latents_cache, emb_cache, sample_emb, emit):
    import torch

    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save({"latents": latents_cache, "embeds": emb_cache, "sample": sample_emb}, cache_file)
        emit(evt("log", level="info", message="Latents+text cache saved (later runs are faster)"))
    except Exception as e:
        emit(evt("log", level="warn", message=f"cache write failed: {e}"))


def _sample_sigma(device):
    """Logit-normal: concentrates sampling toward the middle of the path (better
    que l'uniforme pour le rectified flow)."""
    import torch

    u = torch.randn(1, device=device)
    return torch.sigmoid(u).item()


def _export_lora(transformer, out_dir, name, emit, meta=None):
    """Export the LoRA adapter in diffusers format (prefix `transformer.`),
    loadable by ZImagePipeline.load_lora_weights and recent Z-Image loaders."""
    try:
        import torch
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        sd = get_peft_model_state_dict(transformer)
        out = {}
        for k, v in sd.items():
            # peft -> diffusers : `base_model.model.<...>` devient `transformer.<...>`
            k = k.replace("base_model.model.", "")
            if not k.startswith("transformer."):
                k = "transformer." + k
            out[k] = v.detach().to("cpu", torch.float16)
        path = os.path.join(out_dir, name + ".safetensors")
        save_file(out, path, metadata=meta or None)
        emit(evt("log", level="info", message=f"Z-Image LoRA exported: {name}.safetensors"))
        return path
    except Exception as e:
        emit(evt("log", level="warn", message=f"LoRA export failed: {e}"))
        return None


def run_zimage_training(cfg, emit, stop_event):
    import torch
    import torchvision.transforms as T
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
        cfg.mixed_precision, torch.bfloat16
    )

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"No images found in {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) in the dataset"))

    buckets = _buckets_for_resolution(cfg.resolution)
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])  # -> [-1, 1]

    # Persistent cache: if the latents + embeddings of THIS dataset/resolution/token
    # already exist on disk, we skip encoding AND loading Qwen (8 GB).
    cache_file = os.path.join(cfg.output_dir, ".soma_cache", f"zimage_{_cache_key(data, cfg)}.pt")
    cached = _load_cache(cache_file, emit) if os.path.isfile(cache_file) else None

    pipe = _load_pipeline(cfg, dtype, emit, need_te=(cached is None))

    if cached is not None:
        emit(evt("log", level="info", message="Cache found → latents + embeddings reused"))
        latents_cache, emb_cache, sample_emb = cached
        if getattr(pipe, "vae", None) is not None:
            pipe.vae.to("cpu", dtype=dtype)  # kept warm for the preview
    else:
        # 1) VAE latents (the VAE then stays warm on CPU for the preview)
        emit(evt("log", level="info", message="Pre-computing latents VAE…"))
        vae = pipe.vae
        scaling = vae.config.scaling_factor
        shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
        vae.to(device, dtype=torch.float32)  # encodage en fp32 = stable
        latents_cache = []  # (latent_cpu, caption)
        with torch.no_grad():
            for path, caption in data:
                img, W, H = _load_bucketed(path, buckets)
                px = norm(img).unsqueeze(0).to(device, dtype=torch.float32)
                raw = vae.encode(px).latent_dist.sample()
                x1 = (raw - shift) * scaling
                latents_cache.append((x1.squeeze(0).to("cpu", torch.float32), caption))
        vae.to("cpu", dtype=dtype)  # frees the GPU, keeps the VAE for the preview
        gc.collect()
        torch.cuda.empty_cache()

        # 2) text embeddings (Qwen), then UNLOAD the text encoder
        emit(evt("log", level="info", message="Pre-computing text embeddings (Qwen)…"))
        pipe.text_encoder.to(device)
        emb_cache = []
        default_cap = f"a photo of {cfg.instance_token} person"
        with torch.no_grad():
            for _, caption in data:
                pe, _ = pipe.encode_prompt(
                    caption or default_cap, device=device, do_classifier_free_guidance=False
                )
                emb_cache.append(pe[0].to("cpu", dtype))
            # preview prompt embedding (allows sampling WITHOUT reloading Qwen)
            sample_pe, _ = pipe.encode_prompt(
                cfg.sample_prompt, device=device, do_classifier_free_guidance=False
            )
            sample_emb = sample_pe[0].to("cpu", dtype)
        pipe.text_encoder.to("cpu")
        del pipe.text_encoder
        pipe.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()

        _save_cache(cache_file, latents_cache, emb_cache, sample_emb, emit)

    # ------------------------------------------------------------------
    # 3) TRANSFORMER + LoRA on the GPU (only resident model during the loop)
    # ------------------------------------------------------------------
    emit(evt("log", level="info", message="Transformer 6B -> GPU + LoRA…"))
    transformer = pipe.transformer
    from quant import is_quantized

    if is_quantized(getattr(cfg, "precision", "bf16")):
        transformer.to(device)  # triggers the real nf4/int8 quantization on GPU
    else:
        transformer.to(device, dtype=dtype)
    transformer.requires_grad_(False)
    lora = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        init_lora_weights="gaussian",
        target_modules=_LORA_TARGETS,
    )
    transformer = get_peft_model(transformer, lora)
    pipe.transformer = transformer
    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    params = [p for p in transformer.parameters() if p.requires_grad]

    try:
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(params, lr=cfg.learning_rate)
        emit(evt("log", level="info", message="Optimizer: AdamW8bit (bitsandbytes)"))
    except Exception:
        opt = torch.optim.AdamW(params, lr=cfg.learning_rate)
        emit(evt("log", level="info", message="Optimizer: AdamW (fallback)"))

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
            cap_emb = emb_cache[i].to(device, dtype)                 # [seq,2560]
            x0 = torch.randn_like(x1)
            sigma = _sample_sigma(device)
            noisy = sigma * x0 + (1.0 - sigma) * x1
            model_t = torch.tensor([1.0 - sigma], device=device, dtype=dtype)

            # x = liste de tenseurs (C, F=1, H, W) ; cap_feats = liste [seq, 2560]
            x_list = list(noisy.unsqueeze(2).unbind(dim=0))
            pred = transformer(x_list, model_t, [cap_emb], return_dict=False)[0]
            pred = torch.stack(pred, dim=0).squeeze(2)  # [1,16,h,w]

            target = x1 - x0  # the model predicts natively (data - noise)
            loss = torch.nn.functional.mse_loss(pred.float(), target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()

            emit(
                evt("step", step=step, total_steps=cfg.max_steps,
                    loss=round(loss.item(), 4), lr=cfg.learning_rate,
                    secs=round(time.time() - t0, 1))
            )
            if step % cfg.sample_every == 0 or step == cfg.max_steps:
                _sample(pipe, transformer, sample_emb, cfg, step, emit, device, dtype)
        if stop_event.is_set():
            break

    out = os.path.join(cfg.output_dir, cfg.project_name)
    os.makedirs(out, exist_ok=True)
    transformer.save_pretrained(out)  # format PEFT (rechargeable)
    from families import get_family, soma_meta

    lora_path = _export_lora(transformer, out, cfg.project_name, emit,
                             meta=soma_meta(cfg, get_family(getattr(cfg, "arch", "zimage")), step))
    emit(evt("status", state="done", step=step,
             secs=round(time.time() - t0, 1), output=out, comfyui=lora_path))


def _sample(pipe, transformer, sample_emb, cfg, step, emit, device, dtype):
    """Turbo preview (few steps, guidance 0). We move the VAE back to the GPU and
    pass the ALREADY-computed prompt embedding -> no need to reload Qwen."""
    import torch

    emit(evt("status", state="sampling", step=step))
    try:
        transformer.eval()
        pipe.vae.to(device, dtype=dtype)
        with torch.no_grad():
            image = pipe(
                prompt=None,
                prompt_embeds=[sample_emb.to(device, dtype)],
                height=cfg.resolution,
                width=cfg.resolution,
                num_inference_steps=9,
                guidance_scale=0.0,  # Turbo : pas de CFG
                generator=torch.Generator(device=device).manual_seed(cfg.seed),
            ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        emit(
            evt("sample", step=step, total_steps=cfg.max_steps, placeholder=False,
                image="data:image/png;base64," + b64, prompt=cfg.sample_prompt,
                sharpness=round(step / cfg.max_steps, 3))
        )
    except Exception as e:
        emit(evt("log", level="warn", message=f"sample failed: {e}"))
    finally:
        pipe.vae.to("cpu")
        transformer.train()
        torch.cuda.empty_cache()
        emit(evt("status", state="training", step=step))
