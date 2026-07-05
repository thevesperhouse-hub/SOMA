"""Schéma de configuration d'un entraînement (validé par pydantic)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    # Projet / données
    project_name: str = Field("my-character", description="Nom du LoRA produit")
    arch: str = "sdxl"  # sdxl | zimage — choisit le trainer enfichable
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # Z-Image en fichiers séparés (format ComfyUI) : si base_model est un DiT
    # local .safetensors, on assemble la pipeline avec ce VAE + ce text encoder.
    # Vides => auto-détection dans l'arbo ComfyUI (models/vae, models/text_encoders).
    zimage_vae: str = ""
    zimage_text_encoder: str = ""
    dataset_dir: str = ""  # dossier d'images (+ .txt de caption optionnels)
    instance_token: str = "ohwx"  # token unique du personnage
    output_dir: str = "output"

    # Hyperparamètres LoRA
    resolution: int = 1024
    rank: int = 16
    alpha: int = 16
    learning_rate: float = 1e-4
    max_steps: int = 1200
    batch_size: int = 1

    # Mémoire / perf (pensé 16 GB)
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # bf16 | fp16 | fp32 (calcul)
    precision: str = "bf16"        # bf16 | int8 | nf4 (quantization des poids du modèle)

    # Échantillonnage live ("watch it learn")
    sample_every: int = 100
    sample_prompt: str = "a portrait photo of ohwx person, natural light, sharp focus"
    seed: int = 42

    # Mode démo : pas de GPU / modèle requis, courbe + samples simulés
    simulate: bool = True


class CaptionConfig(BaseModel):
    dataset_dir: str = ""
    instance_token: str = "ohwx"
    model_id: str = "fancyfeast/llama-joycaption-beta-one-hf-llava"
    prompt: str = "Write a detailed description for this image."
    max_new_tokens: int = 256
    prepend_token: bool = True   # préfixe le token au caption (LoRA de perso)
    overwrite: bool = False      # re-captionner même si un .txt existe
    output_dir: str = ""         # vide = à côté des images (recommandé pour l'auto-détection à l'entraînement)


class CaptionSave(BaseModel):
    path: str   # chemin de l'IMAGE ; le .txt est écrit à côté (ou dans output_dir)
    text: str
    output_dir: str = ""
