"""Shared quantization (bitsandbytes) for the transformer backends.

Centralizes 3 things (used by Z-Image, Flux, etc.):
  - build the BitsAndBytesConfig for the requested precision
  - work around the diffusers `from_single_file` + bnb bug (pre_quantized=True by
    default even though we're quantizing a "fresh" bf16 .safetensors)
  - note: after a quantized from_single_file load the model is on CPU, so
    from_single_file must be given device="cuda" to run the real GPU quantization.

precision: "bf16" (no quant) | "int8" (bnb 8-bit) | "nf4" (bnb 4-bit).
"""

_patched = False


def bnb_config(precision: str):
    """Return a BitsAndBytesConfig or None (bf16 = no quantization)."""
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
    """Force pre_quantized=False on the quantizer created by from_single_file, otherwise
    it requires an already-quantized checkpoint (`bitsandbytes__*`). Idempotent."""
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
