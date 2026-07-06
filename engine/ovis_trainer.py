"""Real LoRA training for Ovis-Image — diffusers + peft, QLoRA nf4.

Ovis-Image = Flux arch (packed 2×2 → 64 channels, KL VAE 16ch, 2D ids) BUT text encoded
par **Qwen3** (chat template) et **sans guidance**. Distributed as a diffusers repo →
from_pretrained(subfolder). base_model = repo/folder (required, no reliable default).

Verified (pipeline_ovis_image.py) : message chat = system_prompt + caption ; embeds =
`Qwen3(ids, mask).last_hidden_state * mask` then we DROP the first 28 tokens (prefix);
text_ids = zeros[seq,3] ; flow standard → timestep = sigma (le pipeline passe /1000),
CIBLE = x0 - x1 ; forward = transformer(hidden_states[B,seq,64], timestep, encoder_hidden_states,
txt_ids, img_ids).
"""
import gc
import os
import random
import time

from captioner import clean_path
from events import evt
from train_utils import make_lr_scheduler
from flux_trainer import _export_lora, _latent_image_ids, _load_square, _pack_latents, _sample_sigma
from real_trainer import _list_dataset

_SYSTEM_PROMPT = ("Describe the image by detailing the color, quantity, text, shape, size, "
                  "texture, spatial relationships of the objects and background: ")
_DROP = 28
_MAXLEN = 256 + _DROP
_LORA_TARGETS = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out", "proj_out", "proj_mlp",
]


def run_ovis_training(cfg, emit, stop_event, family=None):
    import torch
    import torchvision.transforms as T
    from diffusers import AutoencoderKL, OvisImageTransformer2DModel
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen3Model

    from quant import bnb_config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    precision = getattr(cfg, "precision", "nf4") or "nf4"
    src = clean_path(cfg.base_model)
    if not src:
        raise RuntimeError("Ovis-Image: provide the model's diffusers repo/folder (base_model).")

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"No images in {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) — Ovis-Image QLoRA ({precision}) from {src}"))

    res = int(cfg.resolution)
    if res % 16 != 0:
        res = (res // 16) * 16
    h = w = res // 8
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    # ---------------- 1) cache latents (VAE KL 16ch, packed 64) ----------------
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
            packed = _pack_latents(x1, 1, 16, h, w).squeeze(0)
            latents_cache.append((packed.to("cpu", dtype), caption))
    img_ids = _latent_image_ids(h // 2, w // 2, device, dtype)
    del vae
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 2) cache embeddings texte (Qwen3, chat template) ----------------
    emit(evt("log", level="info", message="Pre-computing text embeddings (Qwen3)…"))
    tok = AutoTokenizer.from_pretrained(src, subfolder="tokenizer")
    bnb_te = bnb_config(precision if precision != "bf16" else "nf4")
    tekw = dict(subfolder="text_encoder", torch_dtype=dtype)
    if bnb_te is not None:
        tekw["quantization_config"] = bnb_te
        tekw["device_map"] = {"": 0}
    te = Qwen3Model.from_pretrained(src, **tekw)
    if bnb_te is None:
        te = te.to(device)
    te.eval()
    default_cap = f"a photo of {cfg.instance_token} person"
    emb_cache = []
    with torch.no_grad():
        for _, caption in data:
            msg = tok.apply_chat_template(
                [{"role": "user", "content": _SYSTEM_PROMPT + (caption or default_cap)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            toks = tok(msg, padding="max_length", truncation=True, max_length=_MAXLEN,
                       return_tensors="pt", add_special_tokens=False).to(device)
            emb = te(input_ids=toks.input_ids, attention_mask=toks.attention_mask).last_hidden_state
            emb = emb * toks.attention_mask[..., None]
            emb = emb[:, _DROP:, :]  # drops the system prefix
            emb_cache.append(emb.to("cpu", dtype))
    txt_ids = torch.zeros(emb_cache[0].shape[1], 3, device=device, dtype=dtype)
    del te
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 3) transformer nf4 + QLoRA ----------------
    emit(evt("log", level="info", message=f"DiT Ovis-Image ({precision})…"))
    bnb = bnb_config(precision)
    tkw = dict(subfolder="transformer", torch_dtype=dtype)
    if bnb is not None:
        tkw["quantization_config"] = bnb
        tkw["device_map"] = {"": 0}  # nf4 quant on GPU
    transformer = OvisImageTransformer2DModel.from_pretrained(src, **tkw)
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
            emb = emb_cache[i].to(device, dtype)
            x0 = torch.randn_like(x1)
            sigma = _sample_sigma()
            noisy = (1.0 - sigma) * x1 + sigma * x0
            tstep = torch.tensor([sigma], device=device, dtype=dtype)

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
                             meta=soma_meta(cfg, get_family(getattr(cfg, "arch", "ovis")), step))
    emit(evt("status", state="done", step=step, secs=round(time.time() - t0, 1),
             output=out, comfyui=lora_path))
