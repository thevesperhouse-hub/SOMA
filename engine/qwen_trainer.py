"""Vrai entraînement LoRA Qwen-Image — diffusers + peft, QLoRA nf4 (PAS kohya).

Qwen-Image = MMDiT ~20B, objectif flow-matching, texte encodé par Qwen2.5-VL-7B.
Sur 16 Go : DiT bf16 (~40 Go) -> QUANTIFIÉ en nf4 + gradient checkpointing + on
cache latents & embeddings texte puis on décharge VAE/Qwen2.5-VL (ne reste que le
DiT nf4).

Fichiers LOCAUX (ComfyUI) : DiT (qwen_image_edit_*_bf16 de préférence, sinon
fp8_e4m3fn casté en bf16), VAE (qwen_image_vae.safetensors = AutoencoderKLQwenImage
3D). Le TEXT ENCODER Qwen2.5-VL est chargé depuis le repo HF (le fichier ComfyUI
local est dans un layout non-HF ; l'embedding est caché une fois puis déchargé).
Configs : DiT embarqué (model_configs/qwen_transformer), VAE via repo Qwen/Qwen-Image.

API vérifiée par lecture de pipeline_qwenimage.py / transformer_qwenimage.py :
  - encode VAE : image 5D [B,3,1,H,W] -> latent [B,16,1,H',W'] ; normalisation
    model = (raw - latents_mean) / latents_std ; on aplatit T=1 puis PACK 2×2 (64 ch).
  - texte : template système + drop des 34 premiers tokens, dernier hidden state,
    masqué -> (embeds [B,seq,3584], mask [B,seq]).
  - transformer(hidden_states, timestep=sigma, encoder_hidden_states,
    encoder_hidden_states_mask, img_shapes=[[(1,H'/2,W'/2)]], guidance=None) ;
    Qwen-Image N'EST PAS guidance-distillé -> guidance=None. CIBLE = x0 - x1.
"""
import gc
import hashlib
import os
import time

from captioner import clean_path
from events import evt
# helpers génériques partagés avec le trainer Flux (packing 2×2, crop carré,
# sigma logit-normal, export LoRA préfixé "transformer.")
from flux_trainer import _export_lora, _load_square, _pack_latents, _sample_sigma
from real_trainer import _list_dataset

_CFG_DIR = os.path.join(os.path.dirname(__file__), "model_configs", "qwen_transformer")
_VAE_CONFIG_REPO = "Qwen/Qwen-Image"  # config AutoencoderKLQwenImage (subfolder vae)
_TE_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"  # text encoder + tokenizer (HF, correct)
# gabarit système exact du pipeline diffusers + nb de tokens de préfixe à jeter
_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
_PROMPT_DROP_IDX = 34
_TOK_MAX = 1024

# LoRA sur les projections d'attention jointe (image + texte) — set standard et sûr.
_LORA_TARGETS = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
]


# ------------------------------------------------------------------ composants
_keys_patched = False


def _patch_qwen_comfyui_keys():
    """Les checkpoints ComfyUI Qwen ont le préfixe `model.diffusion_model.` mais le
    mapping single_file de diffusers pour QwenImage est IDENTITY (ne le retire pas) →
    toutes les clés seraient ignorées et le modèle resterait sur meta. On strip le
    préfixe (no-op si absent → sûr aussi pour un single-file au format diffusers)."""
    global _keys_patched
    if _keys_patched:
        return
    from diffusers.loaders.single_file_model import SINGLE_FILE_LOADABLE_CLASSES

    import torch

    def _map(checkpoint, **kw):
        out = {}
        for k, v in checkpoint.items():
            # bitsandbytes nf4 exige du 16/32-bit -> caste les poids fp8 (ComfyUI) en bf16
            if getattr(v, "dtype", None) == torch.float8_e4m3fn:
                v = v.to(torch.bfloat16)
            out[k.replace("model.diffusion_model.", "")] = v
        return out

    entry = SINGLE_FILE_LOADABLE_CLASSES.get("QwenImageTransformer2DModel")
    if entry is not None:
        entry["checkpoint_mapping_fn"] = _map
    _keys_patched = True


def _load_transformer(dit_path, precision, cache_dir, emit):
    """DiT Qwen-Image en nf4 (ou bf16). Même cache disque que Flux : quantifié 1×
    (fp8/bf16 -> bf16 -> nf4) puis relu instantanément aux runs suivants."""
    import torch
    from diffusers import QwenImageTransformer2DModel

    from quant import bnb_config, is_quantized, patch_single_file_fresh_quant

    device = "cuda"
    bnb = bnb_config(precision)
    if bnb is None:  # bf16 : ne tient PAS sur 16 Go, mais on laisse le choix
        emit(evt("log", level="info", message="DiT Qwen-Image bf16 (from_single_file)…"))
        tf = QwenImageTransformer2DModel.from_single_file(
            dit_path, config=_CFG_DIR, torch_dtype=torch.bfloat16
        )
        return tf.to(device, dtype=torch.bfloat16)

    key = hashlib.sha1(f"{dit_path}|{os.path.getmtime(dit_path)}|{precision}".encode()).hexdigest()[:12]
    nf4_dir = os.path.join(cache_dir, f"qwen_{precision}_{key}")
    if os.path.isdir(nf4_dir):
        emit(evt("log", level="info", message=f"DiT Qwen-Image {precision} en cache → chargement rapide…"))
        try:
            tf = QwenImageTransformer2DModel.from_pretrained(nf4_dir, torch_dtype=torch.bfloat16)
            return tf.to(device)
        except Exception as e:
            emit(evt("log", level="warn", message=f"cache nf4 illisible ({e}) → re-quantization"))

    emit(evt("log", level="info", message=f"DiT Qwen-Image → {precision} (lecture ~40 Go, ~6 min la 1ère fois)…"))
    patch_single_file_fresh_quant()
    _patch_qwen_comfyui_keys()  # strip 'model.diffusion_model.' (checkpoints ComfyUI)
    # torch_dtype=bf16 : caste un checkpoint fp8_e4m3fn en bf16 avant quantization nf4
    # device="cuda" : quantization nf4 sur GPU (sinon CPU = interminable)
    tf = QwenImageTransformer2DModel.from_single_file(
        dit_path, config=_CFG_DIR, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device="cuda",
    )  # déjà sur GPU quantizé (device="cuda") -> PAS de .to() (casse sur tenseurs meta)
    if is_quantized(precision):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tf.save_pretrained(nf4_dir)
            emit(evt("log", level="info", message="DiT nf4 mis en cache (runs suivants rapides)"))
        except Exception as e:
            emit(evt("log", level="warn", message=f"cache nf4 non écrit: {e}"))
    return tf


def _load_vae(vae_path, dtype, emit):
    # AutoencoderKLQwenImage NE supporte PAS from_single_file (pas dans la liste
    # FromOriginalModelMixin) -> on charge depuis le repo (subfolder vae, ~250 Mo, public).
    from diffusers import AutoencoderKLQwenImage

    emit(evt("log", level="info", message=f"VAE Qwen ({_VAE_CONFIG_REPO})…"))
    return AutoencoderKLQwenImage.from_pretrained(
        _VAE_CONFIG_REPO, subfolder="vae", torch_dtype=dtype
    )


def _load_text_encoder(precision, emit):
    """Qwen2.5-VL-7B depuis le repo HF (layout ComfyUI local non compatible HF).
    nf4 pour tenir la VRAM pendant le pré-calcul, puis déchargé."""
    import torch
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    from quant import bnb_config

    emit(evt("log", level="info", message="Text encoder Qwen2.5-VL-7B (HF, nf4)…"))
    tok = AutoTokenizer.from_pretrained(_TE_REPO)
    bnb = bnb_config(precision if precision != "bf16" else "nf4")  # le TE tient rarement en bf16
    kw = dict(torch_dtype=torch.bfloat16)
    if bnb is not None:
        kw["quantization_config"] = bnb
        kw["device_map"] = {"": 0}
    te = Qwen2_5_VLForConditionalGeneration.from_pretrained(_TE_REPO, **kw)
    if bnb is None:
        te = te.to("cuda")
    return te.eval(), tok


def _encode_prompt(te, tok, prompt, device, dtype):
    """Réplique _get_qwen_prompt_embeds du pipeline : gabarit + drop 34 tokens +
    dernier hidden state masqué. Renvoie (embeds [1,seq,3584], mask [1,seq])."""
    import torch

    txt = _PROMPT_TEMPLATE.format(prompt)
    toks = tok([txt], max_length=_TOK_MAX + _PROMPT_DROP_IDX, padding=True,
               truncation=True, return_tensors="pt").to(device)
    out = te(input_ids=toks.input_ids, attention_mask=toks.attention_mask,
             output_hidden_states=True)
    hidden = out.hidden_states[-1]  # [1, L, 3584]
    mask = toks.attention_mask
    valid = int(mask[0].sum().item())
    emb = hidden[0, :valid][_PROMPT_DROP_IDX:]  # jette le préfixe système
    emb = emb.unsqueeze(0).to(dtype)            # [1, seq, 3584]
    m = torch.ones(1, emb.shape[1], dtype=torch.long, device=emb.device)
    return emb, m


def _find_qwen_components(dit_path, cfg):
    """Localise la VAE Qwen dans l'arbo ComfyUI (relatif au DiT)."""
    from zimage_trainer import _auto_component, _find_models_root

    root = _find_models_root(dit_path)
    if root is None:
        raise RuntimeError("Arbo ComfyUI models/ introuvable près du DiT Qwen (vae).")
    vae = clean_path(getattr(cfg, "zimage_vae", "")) or _auto_component(
        root, "vae", "qwen_image_vae.safetensors", ["qwen_image_vae", "qwen"]
    )
    if not vae or not os.path.isfile(vae):
        raise RuntimeError(f"VAE Qwen (qwen_image_vae.safetensors) introuvable dans {root}")
    return vae


def run_qwen_training(cfg, emit, stop_event, family=None):
    import torch
    import torchvision.transforms as T
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    precision = getattr(cfg, "precision", "nf4") or "nf4"

    dit_path = clean_path(cfg.base_model)
    if not (dit_path.lower().endswith((".safetensors", ".ckpt")) and os.path.isfile(dit_path)):
        raise RuntimeError("Qwen-Image : base_model doit pointer un DiT local (qwen_image_*.safetensors).")

    dataset_dir = clean_path(cfg.dataset_dir)
    data = _list_dataset(dataset_dir)
    if not data:
        raise RuntimeError(f"Aucune image dans {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(data)} image(s) — Qwen-Image QLoRA ({precision})"))

    vae_path = _find_qwen_components(dit_path, cfg)

    # VAE Qwen : compression spatiale = 2**len(temperal_downsample) (=8). res doit
    # être divisible par vsf*2 (packing 2×2). On borne d'abord à vsf*2.
    res = int(cfg.resolution)
    cache_dir = os.path.join(cfg.output_dir, ".soma_cache")
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])  # -> [-1,1]

    # ---------------- 1) cache latents (VAE) ----------------
    emit(evt("log", level="info", message="Pré-calcul des latents (VAE Qwen)…"))
    vae = _load_vae(vae_path, torch.float32, emit).to(device)
    vsf = 2 ** len(vae.temperal_downsample)  # 8
    if res % (vsf * 2) != 0:
        res = (res // (vsf * 2)) * (vsf * 2)
    h_lat = res // vsf                        # dims latentes VAE
    z = vae.config.z_dim                       # 16
    lm = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(device, torch.float32)
    ls = torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(device, torch.float32)
    latents_cache = []
    with torch.no_grad():
        for path, caption in data:
            px = norm(_load_square(path, res)).unsqueeze(0).to(device, torch.float32)  # [1,3,H,W]
            px = px.unsqueeze(2)  # [1,3,1,H,W] (VAE 3D attend une dim temporelle)
            raw = vae.encode(px).latent_dist.sample()  # [1,z,1,H',W']
            x1 = (raw - lm) / ls                         # normalisation modèle
            x1 = x1[:, :, 0]                              # aplatit T=1 -> [1,z,H',W']
            packed = _pack_latents(x1, 1, z, h_lat, h_lat).squeeze(0)  # [seq, z*4=64]
            latents_cache.append((packed.to("cpu", dtype), caption))
    # RoPE : structure (frame=1, H'/2, W'/2), constante (résolution fixe)
    img_shapes = [[(1, h_lat // 2, h_lat // 2)]]
    del vae, lm, ls
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- 2) cache embeddings texte (Qwen2.5-VL) ----------------
    emit(evt("log", level="info", message="Pré-calcul des embeddings texte (Qwen2.5-VL)…"))
    te, tok = _load_text_encoder(precision, emit)
    default_cap = f"a photo of {cfg.instance_token}"
    emb_cache = []
    with torch.no_grad():
        for _, caption in data:
            emb, m = _encode_prompt(te, tok, caption or default_cap, device, dtype)
            emb_cache.append((emb.to("cpu"), m.to("cpu")))
    del te
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

    emit(evt("status", state="training", total_steps=cfg.max_steps))
    transformer.train()
    t0 = time.time()
    step = 0
    import random

    idx = list(range(len(latents_cache)))
    while step < cfg.max_steps:
        random.shuffle(idx)
        for i in idx:
            if stop_event.is_set() or step >= cfg.max_steps:
                break
            step += 1
            x1 = latents_cache[i][0].unsqueeze(0).to(device, dtype)  # [1,seq,64]
            emb, m = emb_cache[i]
            emb = emb.to(device, dtype)
            m = m.to(device)
            x0 = torch.randn_like(x1)
            sigma = _sample_sigma()
            noisy = (1.0 - sigma) * x1 + sigma * x0
            tstep = torch.tensor([sigma], device=device, dtype=dtype)

            pred = transformer(
                hidden_states=noisy, timestep=tstep,
                encoder_hidden_states=emb, encoder_hidden_states_mask=m,
                img_shapes=img_shapes, guidance=None, return_dict=False,
            )[0]
            target = x0 - x1  # flow-matching : cible = noise - data
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
                             meta=soma_meta(cfg, get_family(getattr(cfg, "arch", "qwen_image")), step))
    emit(evt("status", state="done", step=step, secs=round(time.time() - t0, 1),
             output=out, comfyui=lora_path))
