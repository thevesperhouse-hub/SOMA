"""Shared training utilities used by every trainer.

Centralizes the standard anti-overfitting / anti-forgetting guards so each per-family
trainer stays thin (adding an arch is still ~1 file):
  - make_lr_scheduler : linear warmup + cosine decay (softer than a constant LR)
  - maybe_drop        : caption dropout (train unconditional -> less token overfit)
  - min_snr_weights   : min-SNR-gamma loss weighting for epsilon/DDPM trainers
"""
import math
import random


def make_lr_scheduler(optimizer, max_steps, warmup_ratio=0.05, min_lr_ratio=0.05):
    """Linear warmup over the first ``warmup_ratio`` of training, then cosine decay
    down to ``min_lr_ratio`` * base_lr. Returns a LambdaLR; call ``.step()`` right
    after each ``optimizer.step()``. Much gentler than a constant LR, which is the
    main cause of a LoRA "frying" the base model.
    """
    import torch

    warmup = max(1, int(max_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def maybe_drop(caption, p):
    """Caption dropout: with probability ``p`` return "" (an unconditional prompt).
    Teaches the LoRA to also work without the caption -> better generalization and
    less overfitting of everything onto the instance token.
    """
    if p and p > 0.0 and random.random() < p:
        return ""
    return caption


def min_snr_weights(scheduler, timesteps, gamma=5.0, prediction_type="epsilon"):
    """min-SNR-gamma loss weighting (Hang et al., 2023) for DDPM / epsilon-style
    training. Down-weights the high-noise steps that otherwise wash out fine detail
    (faces!). Returns a per-sample weight tensor to multiply the per-sample MSE.
    ``gamma <= 0`` disables it (returns ones).
    """
    import torch

    if not gamma or gamma <= 0:
        return torch.ones_like(timesteps, dtype=torch.float32)
    ac = scheduler.alphas_cumprod.to(timesteps.device)[timesteps].float()
    snr = ac / (1.0 - ac).clamp(min=1e-8)
    clipped = snr.clamp(max=gamma)
    if prediction_type == "v_prediction":
        return clipped / (snr + 1.0)
    return clipped / snr.clamp(min=1e-8)
