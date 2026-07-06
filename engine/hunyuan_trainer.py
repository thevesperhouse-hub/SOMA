"""Real LoRA training for HunyuanImage — diffusers + peft, QLoRA nf4.

HunyuanImage = DiT flow-matching (~17B), texte principal **Qwen2.5-VL** (+ ByT5 pour le
quoted-text rendering — IGNORED here, useless for an identity LoRA), VAE
**AutoencoderKLHunyuanImage** (32× compression, 64 latent channels, 4D unpacked latents,
in=64). Repo diffusers → from_pretrained(subfolder). base_model default tencent/HunyuanImage-2.1.

Verified (pipeline_hunyuanimage.py / transformer_hunyuanimage.py) : Qwen2.5-VL avec gabarit
system + drop 34 tokens + `hidden_states[-3]` (skip_layer=2) + mask ; use_meanflow=False →
`timestep_r=None` ; guidance_embeds=False → `guidance=None` ; glyph off → `encoder_hidden_states_2
=None`. Flow standard : timestep = sigma*1000, CIBLE = x0 - x1. forward = transformer(
hidden_states[B,64,H,W], timestep, encoder_hidden_states, encoder_attention_mask).
"""
import gc
import os
import random
import time

from captioner import clean_path
from events import evt
from flux_trainer import _export_lora, _load_square, _sample_sigma
from real_trainer import _list_dataset

_HY_REPO = "tencent/HunyuanImage-2.1"
_TEMPLATE = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
             "texture, quantity, text, spatial relationships of the objects and background:"
             "<|im_end|>\n<|im_start|>user\n{}<|im_end|>")
_DROP = 34
_SKIP = 2
_TOK_MAX = 256
_LORA_TARGETS = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
]


def _resolve(base_model):
    b = clean_path(base_model)
    return b if b else _HY_REPO


def run_hunyuan_training(cfg, emit, stop_event, family=None):
    import torch
    import torchvision.transforms as T
    from diffusers import AutoencoderKLHunyuanImage, HunyuanImageTransformer2DModel
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    from quant import bnb_config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    precision = getattr(cfg, "precision", "nf4") or "nf4"
    src = _resolve(cfg.base_model)

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"No images in {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) — HunyuanImage QLoRA ({precision}) from {src}"))

    res = int(cfg.resolution)
    if res % 64 != 0:
        res = (res // 64) * 64
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    # ---------------- 1) cache latents (VAE Hunyuan 64ch) ----------------
    emit(evt("log", level="info", message="Pre-computing latents (VAE Hunyuan)…"))
    vae = AutoencoderKLHunyuanImage.from_pretrained(src, subfolder="vae", torch_dtype=torch.float32).to(device)
    scaling = getattr(vae.config, "scaling_factor", None) or 1.0
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    latents_cache = []
    with torch.no_grad():
        for path, caption in data:
            px = norm(_load_square(path, res)).unsqueeze(0).to(device, torch.float32)
            enc = vae.encode(px)
            lat = enc.latent_dist.sample() if hasattr(enc, "latent_dist") else (
                enc.latent if hasattr(enc, "latent") else enc[0])
            x1 = (lat - shift) * scaling  # [1,64,h,w]
            latents_cache.append((x1.squeeze(0).to("cpu", dtype), caption))
    del vae
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 2) cache embeddings texte (Qwen2.5-VL) ----------------
    emit(evt("log", level="info", message="Pre-computing text embeddings (Qwen2.5-VL)…"))
    tok = AutoTokenizer.from_pretrained(src, subfolder="tokenizer")
    bnb_te = bnb_config(precision if precision != "bf16" else "nf4")
    tekw = dict(subfolder="text_encoder", torch_dtype=dtype)
    if bnb_te is not None:
        tekw["quantization_config"] = bnb_te
        tekw["device_map"] = {"": 0}
    te = Qwen2_5_VLForConditionalGeneration.from_pretrained(src, **tekw)
    if bnb_te is None:
        te = te.to(device)
    te.eval()
    default_cap = f"a photo of {cfg.instance_token} person"
    emb_cache = []
    with torch.no_grad():
        for _, caption in data:
            txt = _TEMPLATE.format(caption or default_cap)
            toks = tok(txt, max_length=_TOK_MAX + _DROP, padding="max_length", truncation=True,
                       return_tensors="pt").to(device)
            out = te(input_ids=toks.input_ids, attention_mask=toks.attention_mask,
                     output_hidden_states=True)
            emb = out.hidden_states[-(_SKIP + 1)][:, _DROP:]     # [-3], drop prefix
            mask = toks.attention_mask[:, _DROP:]
            emb_cache.append((emb.to("cpu", dtype), mask.to("cpu")))
    del te
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 3) transformer nf4 + QLoRA ----------------
    emit(evt("log", level="info", message=f"DiT HunyuanImage ({precision})…"))
    bnb = bnb_config(precision)
    tkw = dict(subfolder="transformer", torch_dtype=dtype)
    if bnb is not None:
        tkw["quantization_config"] = bnb
        tkw["device_map"] = {"": 0}  # nf4 quant on GPU
    transformer = HunyuanImageTransformer2DModel.from_pretrained(src, **tkw)
    if bnb is None:
        transformer = transformer.to(device)
    transformer.requires_grad_(False)
    lora = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, init_lora_weights="gaussian",
                      target_modules=_LORA_TARGETS)
    transformer = get_peft_model(transformer, lora)
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
            x1 = latents_cache[i][0].unsqueeze(0).to(device, dtype)  # [1,64,h,w]
            emb, mask = emb_cache[i]
            emb = emb.to(device, dtype)
            mask = mask.to(device)
            x0 = torch.randn_like(x1)
            sigma = _sample_sigma()
            noisy = (1.0 - sigma) * x1 + sigma * x0
            tstep = torch.tensor([sigma * 1000.0], device=device, dtype=dtype)

            pred = transformer(
                hidden_states=noisy, timestep=tstep,
                encoder_hidden_states=emb, encoder_attention_mask=mask,
                timestep_r=None, encoder_hidden_states_2=None, guidance=None,
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
                             meta=soma_meta(cfg, get_family(getattr(cfg, "arch", "hunyuanimage")), step))
    emit(evt("status", state="done", step=step, secs=round(time.time() - t0, 1),
             output=out, comfyui=lora_path))
