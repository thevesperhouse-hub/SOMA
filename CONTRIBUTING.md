# Contributing to SOMA

Thanks for wanting to make SOMA better! The project is built to be **easy to extend** — most
notably, adding a whole new model architecture is roughly **one file**. This guide gets you set up
and walks through the most common contribution.

## Dev setup

```bash
git clone https://github.com/thevesperhouse-hub/SOMA.git && cd SOMA
python engine/bootstrap.py        # venv + torch (cu128) + engine deps
# web UI (optional):
npm install
npm run dev                       # Vite on :1420, talks to the engine on :8765
```

Run the engine directly:

```bash
engine/.venv/bin/python engine/cli.py           # TUI launcher
engine/.venv/bin/python engine/cli.py archs     # list architectures
engine/.venv/bin/python engine/cli.py train --demo --steps 200   # UI/pipeline, no GPU
```

- **Python**: 3.12, style close to the surrounding code (compact, commented where it's non-obvious).
- **transformers stays on v4.x** (`>=4.50,<5`) — v5 breaks many community models.
- Keep new deps minimal; heavy imports go **inside** functions so `--demo`/`serve` stay light.

## 🧩 Add a new architecture (the fun one)

Every model is declared once in `engine/families.py` and handled by a pluggable trainer. To add one:

### 1. Study the reference

Read the model's `diffusers` pipeline + transformer source. You need exactly four things:

| What | Where to look |
|---|---|
| **Text embeddings** | `_get_*_prompt_embeds` / `encode_prompt` in the pipeline |
| **Transformer forward** | the `self.transformer(...)` call in the denoising loop (arg names!) |
| **VAE normalization** | how latents are scaled (`scaling_factor`, `shift_factor`, `latents_mean/std`) |
| **Target sign & timestep** | is the model output negated before the scheduler? is `timestep` sigma or sigma×1000? |

### 2. Write `engine/<arch>_trainer.py`

Copy the closest existing trainer and adapt those four things:

- **Flux-family** (packed 2×2 → 64ch, `img_ids`/`txt_ids`): start from `flux_trainer.py` / `chroma_trainer.py` / `bria_trainer.py`.
- **Unpacked DiT** (4D latents, patchify internal): start from `lumina2_trainer.py` / `sana_trainer.py`.
- **UNet epsilon** (SDXL-style): start from `real_trainer.py` / `sd15_trainer.py`.
- **Repo-based** (not a single-file checkpoint): start from `sd3_trainer.py` (loads components via `from_pretrained(subfolder=…)`).

The common shape: cache latents (VAE) → cache text embeddings → load the DiT in nf4 (QLoRA) → training loop → `_export_lora`.

### 3. Wire it up (two small edits)

```python
# engine/families.py — add to FAMILIES
{"id": "mymodel", "label": "My Model", "backend": "mymodel", "prediction": "flow",
 "resolution": 1024, "default_base": "org/My-Model", "prompt_hint": "a photo of <token>",
 "params_b": 6.0, "quantizable": True},
```

```python
# engine/trainer.py — add to the dispatch
elif fam["backend"] == "mymodel":
    from mymodel_trainer import run_mymodel_training
    run_mymodel_training(self.cfg, self.emit, self._stop_evt, family=fam)
```

### 4. Validate without a full run

Confirm your LoRA target module names actually exist (this catches fused-projection surprises):

```python
from accelerate import init_empty_weights
from diffusers import MyModelTransformer2DModel
with init_empty_weights():
    m = MyModelTransformer2DModel()
names = [n for n, _ in m.named_modules()]
print({t: any(n.endswith("." + t) for n in names) for t in ["to_q", "to_k", "to_v", "to_out.0"]})
```

Then a real 2-step smoke test if you have the weights. Open the PR even if you can't run the big
models — structural validation + a clear description is enough to review.

## Pull requests

- One architecture (or one fix) per PR, with a short note on what you verified.
- Match the existing code style; comment the *why* for anything non-obvious (a timestep convention, a target sign…).
- New architectures that aren't yet integrated are good candidates for `good first issue`.

## Reporting bugs

Open an issue with: the command / config, the arch, your GPU + VRAM, and the full traceback. For training
issues, the step losses and whether it's `--demo` or a real run help a lot.

Happy training. 🔥
