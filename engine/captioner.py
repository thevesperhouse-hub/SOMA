"""Dataset captioning via JoyCaption (uncensored VLM, designed for
diffusion datasets). Loaded in 4-bit (bitsandbytes) then unloaded
immediately. Writes an <image>.txt file per image.

Pattern VRAM transitoire : load -> caption tout -> del + empty_cache.
"""
import gc
import glob
import os
import threading
import traceback

from events import evt

_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
JOYCAPTION = "fancyfeast/llama-joycaption-beta-one-hf-llava"


def clean_path(p):
    """Clean a pasted path: spaces + quotes (Windows 'Copy path')."""
    return (p or "").strip().strip('"').strip("'").strip()


def _list_images(d):
    d = clean_path(d)
    return [p for p in sorted(glob.glob(os.path.join(d, "*"))) if p.lower().endswith(_EXTS)]


def caption_path(image_path, output_dir=""):
    """.txt path for an image: in output_dir if provided, otherwise next to
    the image (recommended: training detects captions placed alongside)."""
    out_dir = clean_path(output_dir)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
    return os.path.splitext(image_path)[0] + ".txt"


class CaptionJob(threading.Thread):
    def __init__(self, cfg, emit):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.emit = emit
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        try:
            self.emit(evt("status", state="captioning", config=self.cfg.model_dump()))
            run_captioning(self.cfg, self.emit, self._stop_evt)
        except Exception as e:  # ne jamais crasher le serveur
            self.emit(evt("status", state="error", message=str(e)))
            self.emit(evt("log", level="error", message=traceback.format_exc()))


# The captioning model is HEAVY (~15-24s to load, ~6 GB VRAM). We keep it
# CACHED in memory between runs: no reload, no VRAM thrash
# (repeated alloc/free that disturbed browser compositing = flicker).
# Freed by clear_model_cache() (called before a training to release the VRAM).
_MODEL_CACHE: dict = {}


def _get_caption_model(model_id, emit):
    cached = _MODEL_CACHE.get(model_id)
    if cached is not None:
        emit(evt("log", level="info", message="Captioning model already in memory (reused)"))
        return cached
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

    emit(evt("log", level="info", message=f"Loading {model_id} (4-bit)…"))
    proc = AutoProcessor.from_pretrained(model_id)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        # NE PAS quantizer le vision tower (SigLIP) ni le projecteur (leur attention
        # F.multi_head_attention_forward casse sur des poids 4-bit).
        llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    _MODEL_CACHE[model_id] = (proc, model)
    return proc, model


def clear_model_cache():
    """Free the captioner's VRAM (call before a training)."""
    import gc

    _MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_captioning(cfg, emit, stop_event):
    import torch
    from PIL import Image

    dataset_dir = clean_path(cfg.dataset_dir)
    images = _list_images(dataset_dir)
    if not images:
        raise RuntimeError(f"No images in {dataset_dir!r}")
    emit(evt("log", level="info", message=f"{len(images)} image(s) to caption"))

    model_id = cfg.model_id or JOYCAPTION
    proc, model = _get_caption_model(model_id, emit)  # loaded once then kept in cache

    token = (cfg.instance_token or "").strip()
    instruction = cfg.prompt or "Write a detailed description for this image."
    emit(evt("status", state="captioning", total=len(images)))
    written = 0
    try:
        for i, path in enumerate(images, 1):
            if stop_event.is_set():
                emit(evt("status", state="stopped"))
                break
            txt_path = caption_path(path, getattr(cfg, "output_dir", ""))
            if os.path.exists(txt_path) and not cfg.overwrite:
                emit(evt("caption", index=i, total=len(images),
                         file=os.path.basename(path), text="(already present)", skipped=True))
                continue
            # "in progress" event: the hero shows the worked image BEFORE generation
            # ("thinking…" state), then the caption appears on it when ready.
            emit(evt("caption", index=i, total=len(images),
                     file=os.path.basename(path), text="", skipped=False))
            image = Image.open(path).convert("RGB")
            convo = [
                {"role": "system", "content": "You are a helpful image captioner."},
                {"role": "user", "content": instruction},
            ]
            convo_string = proc.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=True
            )
            inputs = proc(text=[convo_string], images=[image], return_tensors="pt").to("cuda")
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False)
            caption = proc.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            if cfg.prepend_token and token:
                caption = f"{token}, {caption}"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)
            written += 1
            emit(evt("caption", index=i, total=len(images),
                     file=os.path.basename(path), text=caption, skipped=False))
        emit(evt("status", state="done_caption", written=written, total=len(images)))
    finally:
        # DO NOT unload the model: it stays cached (_MODEL_CACHE) for the
        # following runs. We only free the transient memory.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
