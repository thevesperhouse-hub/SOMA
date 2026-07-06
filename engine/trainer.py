"""Training job run in a thread.

- simulate=True  -> simulated loss curve + samples (no heavy deps, instant demo of
  the live dashboard).
- simulate=False -> real SDXL LoRA training (diffusers + peft), isolated in
  real_trainer.py so torch is only loaded when needed.
"""
import math
import random
import threading
import time
import traceback

from events import evt


class TrainingJob(threading.Thread):
    def __init__(self, cfg, emit):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.emit = emit
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        try:
            self.emit(evt("status", state="starting", config=self.cfg.model_dump()))
            if self.cfg.simulate:
                self._run_sim()
            else:
                self._run_real()
        except Exception as e:  # never crash the server
            self.emit(evt("status", state="error", message=str(e)))
            self.emit(evt("log", level="error", message=traceback.format_exc()))

    # ------------------------------------------------------------------
    # Multi-arch dispatch: each architecture = its own pluggable trainer.
    # SDXL and Z-Image share the dataset/bucketing helpers but have their own
    # objective (epsilon vs flow-matching) and their own pipeline.
    # ------------------------------------------------------------------
    def _run_real(self):
        from families import get_family

        # free the captioner's VRAM (model kept in cache) before training
        try:
            from captioner import clear_model_cache

            clear_model_cache()
        except Exception:
            pass

        fam = get_family(getattr(self.cfg, "arch", "sdxl"))
        self.emit(evt("log", level="info",
                      message=f"Family: {fam['label']} (backend={fam['backend']}, "
                              f"prediction={fam['prediction']})"))
        if fam["backend"] == "zimage":
            from zimage_trainer import run_zimage_training

            run_zimage_training(self.cfg, self.emit, self._stop_evt)
        elif fam["backend"] == "flux":
            from flux_trainer import run_flux_training

            run_flux_training(self.cfg, self.emit, self._stop_evt, family=fam)
        elif fam["backend"] == "qwen":
            from qwen_trainer import run_qwen_training

            run_qwen_training(self.cfg, self.emit, self._stop_evt, family=fam)
        elif fam["backend"] == "chroma":
            from chroma_trainer import run_chroma_training

            run_chroma_training(self.cfg, self.emit, self._stop_evt, family=fam)
        elif fam["backend"] == "lumina2":
            from lumina2_trainer import run_lumina2_training

            run_lumina2_training(self.cfg, self.emit, self._stop_evt, family=fam)
        elif fam["backend"] == "sd15":
            from sd15_trainer import run_sd15_training

            run_sd15_training(self.cfg, self.emit, self._stop_evt, family=fam)
        elif fam["backend"] == "prx":
            from prx_trainer import run_prx_training

            run_prx_training(self.cfg, self.emit, self._stop_evt, family=fam)
        elif fam["backend"] == "sd3":
            from sd3_trainer import run_sd3_training

            run_sd3_training(self.cfg, self.emit, self._stop_evt, family=fam)
        elif fam["backend"] == "sana":
            from sana_trainer import run_sana_training

            run_sana_training(self.cfg, self.emit, self._stop_evt, family=fam)
        elif fam["backend"] == "pixart":
            from pixart_trainer import run_pixart_training

            run_pixart_training(self.cfg, self.emit, self._stop_evt, family=fam)
        elif fam["backend"] == "bria":
            from bria_trainer import run_bria_training

            run_bria_training(self.cfg, self.emit, self._stop_evt, family=fam)
        elif fam["backend"] == "auraflow":
            from auraflow_trainer import run_auraflow_training

            run_auraflow_training(self.cfg, self.emit, self._stop_evt, family=fam)
        elif fam["backend"] == "cogview4":
            from cogview4_trainer import run_cogview4_training

            run_cogview4_training(self.cfg, self.emit, self._stop_evt, family=fam)
        elif fam["backend"] == "ovis":
            from ovis_trainer import run_ovis_training

            run_ovis_training(self.cfg, self.emit, self._stop_evt, family=fam)
        elif fam["backend"] == "kolors":
            from kolors_trainer import run_kolors_training

            run_kolors_training(self.cfg, self.emit, self._stop_evt, family=fam)
        elif fam["backend"] == "hunyuanimage":
            from hunyuan_trainer import run_hunyuan_training

            run_hunyuan_training(self.cfg, self.emit, self._stop_evt, family=fam)
        else:  # SDXL backend: SDXL / Pony / Illustrious / NoobAI (eps & v-pred)
            from real_trainer import run_real_training

            run_real_training(self.cfg, self.emit, self._stop_evt, family=fam)

    # ------------------------------------------------------------------
    # Demo mode: mimics the shape of a real run (decay + shrinking noise +
    # cosine LR) to validate the whole UX without a GPU or model.
    # ------------------------------------------------------------------
    def _run_sim(self):
        cfg = self.cfg
        self.emit(evt("status", state="training", total_steps=cfg.max_steps))
        loss0, floor = 0.16, 0.035
        tau = max(1.0, cfg.max_steps / 3.2)
        t0 = time.time()
        for step in range(1, cfg.max_steps + 1):
            if self._stop_evt.is_set():
                self.emit(evt("status", state="stopped", step=step))
                return
            base = floor + (loss0 - floor) * math.exp(-step / tau)
            # noise that settles as it converges
            jitter = random.uniform(-1, 1) * 0.014 * (0.3 + 0.7 * math.exp(-step / tau))
            loss = max(0.001, base + jitter)
            lr = cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * step / cfg.max_steps))
            self.emit(
                evt(
                    "step",
                    step=step,
                    total_steps=cfg.max_steps,
                    loss=round(loss, 4),
                    lr=lr,
                    secs=round(time.time() - t0, 1),
                )
            )
            if step % cfg.sample_every == 0 or step == cfg.max_steps:
                self.emit(
                    evt(
                        "sample",
                        step=step,
                        total_steps=cfg.max_steps,
                        placeholder=True,  # the front-end draws a procedural preview
                        seed=cfg.seed,
                        prompt=cfg.sample_prompt,
                        sharpness=round(step / cfg.max_steps, 3),
                    )
                )
            time.sleep(0.02)
        self.emit(
            evt("status", state="done", step=cfg.max_steps, secs=round(time.time() - t0, 1))
        )
