"""SOMA — headless CLI (training, server, captioning) for Docker / cloud / scripts.

Reuses EXACTLY the same engine as the UI (config.TrainConfig + trainer.TrainingJob),
with styled terminal output (pure ANSI, zero deps — works in a container).

    python cli.py train  --arch flux --base /models/flux1-dev.safetensors \
                         --dataset /data/mychar --steps 1200 --precision nf4
    python cli.py train  --config run.json
    python cli.py serve  --host 0.0.0.0 --port 8765        # sert l'API + l'UI web
    python cli.py archs
    python cli.py caption --dataset /data/mychar

Sur Vast.ai : `docker run --gpus all -v /data:/data soma train --arch ... ` ; ou
`serve` + expose port 8765 → open the web UI in a browser.
"""
import argparse
import json
import os
import sys
import threading
import time

# ---------------------------------------------------------------- style ANSI
_NO_COLOR = os.environ.get("NO_COLOR") or not sys.stdout.isatty()


def _c(code):
    return "" if _NO_COLOR else code


R = _c("\033[0m"); B = _c("\033[1m"); DIM = _c("\033[2m")
ACC = _c("\033[38;5;79m"); GOOD = _c("\033[38;5;114m")
WARN = _c("\033[38;5;179m"); BAD = _c("\033[38;5;203m"); MUT = _c("\033[38;5;245m")


def _banner(title, sub=""):
    line = "─" * 58
    print(f"\n{ACC}{B}  ▟▛ SOMA{R}{DIM}  ·  {title}{R}")
    if sub:
        print(f"{MUT}  {sub}{R}")
    print(f"{DIM}  {line}{R}")


def _fmt_time(s):
    s = int(s); h, m = divmod(s, 3600); m, s = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


class _Renderer:
    """Turns engine events into terminal output (live bar + logs)."""

    def __init__(self):
        self.total = 0
        self.t0 = time.time()
        self._last_len = 0

    def _clear(self):
        if self._last_len:
            sys.stdout.write("\r" + " " * self._last_len + "\r")
            self._last_len = 0

    def __call__(self, e):
        t = e.get("type")
        if t == "log":
            self._clear()
            lvl = e.get("level", "info")
            col = {"warn": WARN, "error": BAD}.get(lvl, MUT)
            print(f"  {col}·{R} {e.get('message', '')}")
        elif t == "status":
            st = e.get("state")
            if st == "training":
                self.total = e.get("total_steps", 0) or 0
                self.t0 = time.time()
                self._clear()
                print(f"  {ACC}▶ training{R} {DIM}({self.total} steps){R}")
            elif st == "sampling":
                pass
            elif st == "done":
                self._clear()
                secs = e.get("secs", time.time() - self.t0)
                out = e.get("comfyui") or e.get("output") or ""
                print(f"\n  {GOOD}{B}✓ done{R} in {_fmt_time(secs)}")
                if out:
                    print(f"  {GOOD}LoRA →{R} {out}")
            elif st == "error":
                self._clear()
                print(f"\n  {BAD}{B}✗ error{R} {e.get('message', '')}")
        elif t == "step":
            self._render_step(e)
        elif t == "sample":
            self._clear()
            print(f"  {MUT}· preview step {e.get('step')}{R}")

    def _render_step(self, e):
        step = e.get("step", 0); total = e.get("total_steps", self.total) or 1
        loss = e.get("loss", 0.0); secs = e.get("secs", 0.0)
        frac = min(1.0, step / total)
        width = 26
        fill = int(frac * width)
        bar = f"{ACC}{'█' * fill}{DIM}{'░' * (width - fill)}{R}"
        rate = secs / step if step else 0
        eta = _fmt_time(rate * (total - step)) if rate else "—"
        line = (f"  {bar} {B}{int(frac*100):3d}%{R} "
                f"{DIM}{step}/{total}{R}  loss={GOOD}{loss:.4f}{R}  "
                f"{DIM}{rate:.2f}s/it · ETA {eta}{R}")
        self._clear()
        # visible length (without ANSI codes) for clearing
        visible = f"  {'█'*fill}{'░'*(width-fill)} {int(frac*100):3d}% {step}/{total}  loss={loss:.4f}  {rate:.2f}s/it · ETA {eta}"
        self._last_len = len(visible)
        sys.stdout.write(line + "\r"); sys.stdout.flush()


# ---------------------------------------------------------------- commandes
def cmd_archs(_args):
    from families import FAMILIES

    _banner("architectures disponibles", f"{len(FAMILIES)} familles")
    for f in FAMILIES:
        q = f"{GOOD}nf4{R}" if f.get("quantizable") else f"{MUT}bf16{R}"
        print(f"  {B}{f['id']:<16}{R}{DIM}{f['label']:<22}{R} "
              f"backend={f['backend']:<9} {f['prediction']:<12} {q} ~{f.get('params_b','?')}B")
    print()


def _build_cfg(args):
    from config import TrainConfig

    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            data = json.load(fh)
        data["simulate"] = False
        return TrainConfig(**data)
    # from the flags
    fields = dict(
        simulate=False, arch=args.arch, base_model=args.base, dataset_dir=args.dataset,
        project_name=args.project, instance_token=args.token, output_dir=args.output,
        resolution=args.resolution, rank=args.rank, alpha=args.alpha,
        learning_rate=args.lr, max_steps=args.steps, precision=args.precision,
        gradient_checkpointing=not args.no_grad_ckpt, seed=args.seed,
    )
    if args.sample_prompt:
        fields["sample_prompt"] = args.sample_prompt
    if args.zimage_vae:
        fields["zimage_vae"] = args.zimage_vae
    return TrainConfig(**{k: v for k, v in fields.items() if v is not None})


def cmd_train(args):
    from trainer import TrainingJob

    cfg = _build_cfg(args)
    if getattr(args, "demo", False):
        cfg.simulate = True  # simulated curve: see the terminal UI with no GPU or model
    _banner("training", f"{cfg.arch}  ·  {os.path.basename(cfg.base_model) or cfg.base_model}")
    print(f"  {MUT}dataset{R} {cfg.dataset_dir}   {MUT}precision{R} {cfg.precision}   "
          f"{MUT}rank{R} {cfg.rank}   {MUT}steps{R} {cfg.max_steps}   {MUT}res{R} {cfg.resolution}")
    print(f"{DIM}  {'─'*58}{R}")

    render = _Renderer()
    job = TrainingJob(cfg, render)
    job.start()
    try:
        while job.is_alive():
            job.join(timeout=0.5)
    except KeyboardInterrupt:
        print(f"\n  {WARN}· stop requested, finishing cleanly…{R}")
        job.stop()
        job.join()
    print()


def cmd_serve(args):
    import uvicorn

    _banner("serveur", f"API + UI web sur http://{args.host}:{args.port}")
    print(f"  {MUT}web UI{R}: open {ACC}http://<ip>:{args.port}/{R} in a browser")
    print(f"  {MUT}API{R}    : POST /api/train/start · WS /ws\n")
    uvicorn.run("server:app", host=args.host, port=args.port, log_level="warning",
                ws_ping_interval=None, ws_ping_timeout=None)


def cmd_caption(args):
    from captioner import run_captioning
    from config import CaptionConfig

    _banner("captioning", args.dataset)
    render = _Renderer()

    def emit(e):
        if e.get("type") == "caption":
            i, tot = e.get("index", 0), e.get("total", 0)
            render._clear()
            f = e.get("file", "")
            txt = (e.get("text", "") or "").replace("\n", " ")[:60]
            line = f"  {DIM}{i}/{tot}{R} {f}  {MUT}{txt}{R}"
            sys.stdout.write(line + "\r"); sys.stdout.flush()
            render._last_len = len(f) + len(txt) + 12
        else:
            render(e)

    cfg = CaptionConfig(dataset_dir=args.dataset, instance_token=args.token,
                        overwrite=args.overwrite)
    run_captioning(cfg, emit, threading.Event())
    print(f"\n  {GOOD}✓ captions written{R}\n")


def main(argv=None):
    p = argparse.ArgumentParser(prog="soma", description="SOMA — headless LoRA training")
    sub = p.add_subparsers(dest="cmd", required=False)

    pt = sub.add_parser("train", help="run a training")
    pt.add_argument("--config", help="JSON config file (takes priority over the flags)")
    pt.add_argument("--arch", default="sdxl")
    pt.add_argument("--base", default="", help="checkpoint local ou repo HF")
    pt.add_argument("--dataset", default="")
    pt.add_argument("--output", default="output")
    pt.add_argument("--project", default="my-character")
    pt.add_argument("--token", default="ohwx")
    pt.add_argument("--resolution", type=int, default=1024)
    pt.add_argument("--rank", type=int, default=16)
    pt.add_argument("--alpha", type=int, default=16)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--steps", type=int, default=1200)
    pt.add_argument("--precision", default="bf16", choices=["bf16", "int8", "nf4"])
    pt.add_argument("--demo", action="store_true", help="demo mode (simulated curve, no GPU or model)")
    pt.add_argument("--no-grad-ckpt", action="store_true", help="disable gradient checkpointing")
    pt.add_argument("--sample-prompt", default="")
    pt.add_argument("--zimage-vae", default="")
    pt.add_argument("--seed", type=int, default=42)
    pt.set_defaults(func=cmd_train)

    ps = sub.add_parser("serve", help="run the API server + web UI")
    ps.add_argument("--host", default="0.0.0.0")
    ps.add_argument("--port", type=int, default=8765)
    ps.set_defaults(func=cmd_serve)

    pa = sub.add_parser("archs", help="list the architectures")
    pa.set_defaults(func=cmd_archs)

    pc = sub.add_parser("caption", help="caption a dataset")
    pc.add_argument("--dataset", required=True)
    pc.add_argument("--token", default="ohwx")
    pc.add_argument("--overwrite", action="store_true")
    pc.set_defaults(func=cmd_caption)

    args = p.parse_args(argv)
    if not getattr(args, "cmd", None):
        # no subcommand -> interactive launcher (styled TUI)
        try:
            from launcher import run
        except ImportError:
            p.print_help()
            return
        run()
        return
    args.func(args)


if __name__ == "__main__":
    main()
