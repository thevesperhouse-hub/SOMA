"""Quantization partagée (bitsandbytes) pour les backends transformer.

Centralise 3 choses (utilisées par Z-Image, Flux, etc.) :
  - construire la BitsAndBytesConfig selon la précision demandée
  - le contournement du bug diffusers `from_single_file` + bnb (pre_quantized=True
    par défaut alors qu'on quantifie un .safetensors bf16 "frais")
  - rappel : après un load quantizé from_single_file, le modèle est sur CPU ->
    il faut `.to("cuda")` pour déclencher la vraie quantization GPU.

precision : "bf16" (aucune quant) | "int8" (bnb 8-bit) | "nf4" (bnb 4-bit).
"""

_patched = False


def bnb_config(precision: str):
    """Renvoie une BitsAndBytesConfig ou None (bf16 = pas de quantization)."""
    if precision in (None, "", "bf16", "fp16", "fp32"):
        return None
    import torch
    from diffusers import BitsAndBytesConfig

    if precision == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def patch_single_file_fresh_quant():
    """Force pre_quantized=False sur le quantizer créé par from_single_file, sinon
    il exige un checkpoint déjà quantizé (`bitsandbytes__*`). Idempotent."""
    global _patched
    if _patched:
        return
    from diffusers.quantizers.auto import DiffusersAutoQuantizer

    orig = DiffusersAutoQuantizer.from_config.__func__

    def _patched_from_config(cls, quantization_config, **kw):
        q = orig(cls, quantization_config, **kw)
        q.pre_quantized = False
        return q

    DiffusersAutoQuantizer.from_config = classmethod(_patched_from_config)
    _patched = True


def is_quantized(precision: str) -> bool:
    return precision in ("int8", "nf4")
