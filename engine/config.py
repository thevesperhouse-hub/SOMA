"""Training configuration schema (validated by pydantic)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    # Project / data
    project_name: str = Field("my-character", description="Name of the produced LoRA")
    arch: str = "sdxl"  # sdxl | zimage — selects the pluggable trainer
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # Z-Image as separate files (ComfyUI format): if base_model is a local .safetensors
    # DiT, we assemble the pipeline with this VAE + this text encoder.
    # Empty => auto-detection in the ComfyUI tree (models/vae, models/text_encoders).
    zimage_vae: str = ""
    zimage_text_encoder: str = ""
    dataset_dir: str = ""  # image folder (+ optional .txt captions)
    instance_token: str = "ohwx"  # unique character token
    output_dir: str = "output"

    # LoRA hyperparameters
    resolution: int = 1024
    rank: int = 16
    alpha: int = 8          # alpha < rank (scale 0.5) = gentler, less base-model frying
    learning_rate: float = 1e-4
    max_steps: int = 1200
    batch_size: int = 1

    # Anti-overfitting / anti-forgetting guards (see train_utils.py)
    lr_warmup_ratio: float = 0.05   # linear warmup then cosine decay to 5% of base LR
    min_snr_gamma: float = 5.0      # min-SNR-gamma loss weighting (epsilon families); <=0 disables
    caption_dropout: float = 0.1    # fraction of steps trained unconditionally

    # Memory / perf (designed for 16 GB)
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # bf16 | fp16 | fp32 (compute)
    precision: str = "bf16"        # bf16 | int8 | nf4 (model weight quantization)

    # Live sampling ("watch it learn")
    sample_every: int = 100
    sample_prompt: str = "a portrait photo of ohwx person, natural light, sharp focus"
    seed: int = 42

    # Demo mode: no GPU / model required, simulated curve + samples
    simulate: bool = True


class CaptionConfig(BaseModel):
    dataset_dir: str = ""
    instance_token: str = "ohwx"
    model_id: str = "fancyfeast/llama-joycaption-beta-one-hf-llava"
    prompt: str = "Write a detailed description for this image."
    max_new_tokens: int = 256
    prepend_token: bool = True   # prepend the token to the caption (character LoRA)
    overwrite: bool = False      # re-caption even if a .txt already exists
    output_dir: str = ""         # empty = next to the images (recommended for training auto-detection)


class CaptionSave(BaseModel):
    path: str   # IMAGE path; the .txt is written next to it (or in output_dir)
    text: str
    output_dir: str = ""
