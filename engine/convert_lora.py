"""Convertit un adaptateur LoRA PEFT (adapter_model.safetensors) en LoRA format
kohya (.safetensors) chargeable directement dans ComfyUI / A1111.

    python convert_lora.py --adapter "...\\output\\lora01"

Charge l'UNet SDXL (depuis le cache HF) juste pour récupérer le mapping des clés.
"""
import argparse
import os

import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import convert_state_dict_to_kohya
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="dossier avec adapter_model.safetensors")
    ap.add_argument("--base", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    name = os.path.basename(os.path.normpath(a.adapter))
    out = a.out or os.path.join(a.adapter, name + ".safetensors")

    print(f"Chargement de l'UNet ({a.base})…")
    pipe = StableDiffusionXLPipeline.from_pretrained(a.base, torch_dtype=torch.float16)
    print(f"Application de l'adaptateur : {a.adapter}")
    unet = PeftModel.from_pretrained(pipe.unet, a.adapter)

    kohya = convert_state_dict_to_kohya(get_peft_model_state_dict(unet))
    # le wrapper PEFT laisse "base_model_model_" ; ComfyUI veut "lora_unet_".
    kohya = {
        k.replace("base_model_model_", "lora_unet_"): v.detach().to("cpu", torch.float16)
        for k, v in kohya.items()
    }
    save_file(kohya, out)
    print(f"OK -> {out}")
    print(f"     {len(kohya)} tenseurs (format kohya, prêt pour ComfyUI)")


if __name__ == "__main__":
    main()
