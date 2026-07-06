"""Real LoRA training for Stable Diffusion 3.5 — diffusers + peft, QLoRA nf4.

SD3.5 = MMDiT flow-matching, **3 text encoders** : CLIP-L (768) + CLIP-G (1280) +
T5-XXL (4096). Distributed as a diffusers repo → composants chargés via from_pretrained
(subfolder). base_model = HF repo (default stabilityai/stable-diffusion-3.5-medium) ou
dossier diffusers local.

Assemblage embeds (vérifié pipeline_stable_diffusion_3.py) :
  clip = cat([clipL.hidden[-2](768), clipG.hidden[-2](1280)], -1) -> 2048, PAD à 4096,
  puis cat([clip, t5], dim=seq) -> encoder_hidden_states ; pooled = cat([poolL, poolG]) -> 2048.
Flow STANDARD (no negation) : timestep = sigma*1000, CIBLE = x0 - x1.
forward = transformer(hidden_states[B,16,H,W], timestep, encoder_hidden_states,
pooled_projections).
"""
import gc
import os
import random
import time

from captioner import clean_path
from events import evt
from flux_trainer import _export_lora, _load_square, _sample_sigma
from real_trainer import _list_dataset

_SD3_REPO = "stabilityai/stable-diffusion-3.5-medium"
_T5_MAX = 256
_LORA_TARGETS = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
]


def _resolve(base_model):
    b = clean_path(base_model)
    return b if b else _SD3_REPO


def run_sd3_training(cfg, emit, stop_event, family=None):
    import torch
    import torchvision.transforms as T
    from diffusers import AutoencoderKL, SD3Transformer2DModel
    from peft import LoraConfig, get_peft_model
    from transformers import (
        CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast,
    )

    from quant import bnb_config, is_quantized, patch_single_file_fresh_quant  # noqa: F401

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    precision = getattr(cfg, "precision", "nf4") or "nf4"
    src = _resolve(cfg.base_model)

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"No images in {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) — SD3.5 QLoRA ({precision}) from {src}"))

    res = int(cfg.resolution)
    if res % 16 != 0:
        res = (res // 16) * 16
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    # ---------------- 1) cache latents (VAE) ----------------
    emit(evt("log", level="info", message="Pre-computing latents (VAE)…"))
    vae = AutoencoderKL.from_pretrained(src, subfolder="vae", torch_dtype=torch.float32).to(device)
    scaling = vae.config.scaling_factor
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    latents_cache = []
    with torch.no_grad():
        for path, caption in data:
            px = norm(_load_square(path, res)).unsqueeze(0).to(device, torch.float32)
            raw = vae.encode(px).latent_dist.sample()
            x1 = (raw - shift) * scaling  # [1,16,h,w]
            latents_cache.append((x1.squeeze(0).to("cpu", dtype), caption))
    del vae
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 2) cache embeddings texte (CLIP-L + CLIP-G + T5) ----------------
    emit(evt("log", level="info", message="Pre-computing text embeddings (CLIP-L + CLIP-G + T5)…"))
    tok1 = CLIPTokenizer.from_pretrained(src, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(src, subfolder="tokenizer_2")
    tok3 = T5TokenizerFast.from_pretrained(src, subfolder="tokenizer_3")
    te1 = CLIPTextModelWithProjection.from_pretrained(src, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
    te2 = CLIPTextModelWithProjection.from_pretrained(src, subfolder="text_encoder_2", torch_dtype=dtype).to(device).eval()
    te3 = T5EncoderModel.from_pretrained(src, subfolder="text_encoder_3", torch_dtype=dtype).to(device).eval()

    def encode(cap):
        ids1 = tok1(cap, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
        ids2 = tok2(cap, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
        ids3 = tok3(cap, padding="max_length", max_length=_T5_MAX, truncation=True, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            o1 = te1(ids1, output_hidden_states=True)
            o2 = te2(ids2, output_hidden_states=True)
            h = torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1)  # [1,77,2048]
            h = torch.nn.functional.pad(h, (0, 4096 - h.shape[-1]))              # -> 4096
            t5 = te3(ids3)[0]                                                     # [1,seq,4096]
            emb = torch.cat([h, t5], dim=-2)                                      # sequence concat
            pooled = torch.cat([o1[0], o2[0]], dim=-1)                            # [1,2048]
        return emb.to("cpu", dtype), pooled.to("cpu", dtype)

    default_cap = f"a photo of {cfg.instance_token} person"
    emb_cache = [encode(caption or default_cap) for _, caption in data]
    del te1, te2, te3
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 3) transformer nf4 + QLoRA ----------------
    emit(evt("log", level="info", message=f"DiT SD3.5 ({precision})…"))
    bnb = bnb_config(precision)
    tkw = dict(subfolder="transformer", torch_dtype=dtype)
    if bnb is not None:
        tkw["quantization_config"] = bnb
        tkw["device_map"] = {"": 0}  # nf4 quant on GPU
    transformer = SD3Transformer2DModel.from_pretrained(src, **tkw)
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
            x1 = latents_cache[i][0].unsqueeze(0).to(device, dtype)  # [1,16,h,w]
            emb, pooled = emb_cache[i]
            emb = emb.to(device, dtype)
            pooled = pooled.to(device, dtype)
            x0 = torch.randn_like(x1)
            sigma = _sample_sigma()
            noisy = (1.0 - sigma) * x1 + sigma * x0
            tstep = torch.tensor([sigma * 1000.0], device=device, dtype=dtype)  # SD3 : 0..1000

            pred = transformer(
                hidden_states=noisy, timestep=tstep,
                encoder_hidden_states=emb, pooled_projections=pooled,
                return_dict=False,
            )[0]
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
                             meta=soma_meta(cfg, get_family(getattr(cfg, "arch", "sd3")), step))
    emit(evt("status", state="done", step=step, secs=round(time.time() - t0, 1),
             output=out, comfyui=lora_path))
