"""SOMA — launcher terminal (TUI) stylisé, propulsé par `rich`.

Lancé quand `python cli.py` est appelé sans sous-commande : grand logo SOMA en blocs
+ dégradé, panneau système (GPU/VRAM), menu interactif (entraîner / captionner /
UI web / architectures). Le vrai moteur reste le même (config.TrainConfig + TrainingJob).
"""
import base64
import io
import os
import subprocess
import threading
import time
from collections import deque

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

console = Console()

# --------------------------------------------------------------- logo SOMA
_S = ["██████", "██    ", "██████", "    ██", "██████"]
_O = ["██████", "██  ██", "██  ██", "██  ██", "██████"]
_M = ["██   ██", "███ ███", "██ █ ██", "██   ██", "██   ██"]
_A = [" ████ ", "██  ██", "██████", "██  ██", "██  ██"]
_LETTERS = [_S, _O, _M, _A]
# dégradé teal (haut -> bas)
_GRAD = ["#7df0dd", "#43cf9f", "#2bb894", "#1f9c85", "#177c72"]


def _logo_lines():
    return ["  " + "   ".join(letter[row] for letter in _LETTERS) for row in range(5)]


def _logo() -> Text:
    t = Text(justify="center")
    for row, line in enumerate(_logo_lines()):
        t.append(line + "\n", style=f"bold {_GRAD[row]}")
    return t


def _logo_frame(edge: int, reveal: bool) -> Text:
    """Une frame d'anim : `edge` = position du faisceau. reveal=True → écrit
    le logo de gauche à droite ; reveal=False → simple balayage lumineux."""
    t = Text(justify="center")
    for r, line in enumerate(_logo_lines()):
        for x, ch in enumerate(line):
            if ch == " ":
                t.append(" ")
            elif reveal and x > edge:
                t.append(" ")  # pas encore révélé
            elif abs(x - edge) <= 1:
                t.append(ch, style="bold #eafff9")  # faisceau (highlight)
            else:
                t.append(ch, style=f"bold {_GRAD[r]}")
        t.append("\n")
    return t


def _info_panel():
    from families import FAMILIES

    info = Text.from_markup(
        f"  {_gpu_line()}\n"
        f"  [dim]{len(FAMILIES)} architectures[/] · [dim]moteur maison diffusers + peft[/]",
        justify="center",
    )
    return Align.center(Panel(info, border_style="#177c72", width=72, padding=(0, 2)))


_SUBTITLE = "L O R A   T R A I N I N G   S T U D I O"


def intro():
    """Intro animée : le logo s'écrit sous un faisceau, puis sous-titre à la machine
    à écrire. Sautée si pas de vrai terminal (pipe/Docker)."""
    from rich.live import Live

    if not console.is_terminal:
        splash()
        return
    console.clear()
    console.print("\n")
    width = max(len(l) for l in _logo_lines())
    with Live(console=console, refresh_per_second=60, screen=False) as live:
        for edge in range(-2, width + 2):           # écriture gauche→droite
            live.update(Align.center(_logo_frame(edge, reveal=True)))
            time.sleep(0.018)
        for edge in range(-2, width + 3):           # balayage lumineux
            live.update(Align.center(_logo_frame(edge, reveal=False)))
            time.sleep(0.012)
        live.update(Align.center(_logo()))
    # sous-titre machine à écrire
    with Live(console=console, refresh_per_second=60, screen=False) as live:
        for i in range(len(_SUBTITLE) + 1):
            live.update(Align.center(Text(_SUBTITLE[:i], style="dim #7df0dd")))
            time.sleep(0.02)
    console.print()
    console.print(_info_panel())


def _gpu_line() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip().splitlines()[0]
        name, total, used, temp = [x.strip() for x in out.split(",")]
        gb = lambda m: f"{int(float(m)) / 1024:.1f}"
        return f"[bold #7df0dd]{name}[/]  ·  VRAM [bold]{gb(used)}[/]/[dim]{gb(total)} Go[/]  ·  {temp}°C"
    except Exception:
        return "[dim]GPU non détecté (mode démo possible)[/]"


def splash():
    console.clear()
    console.print()
    console.print(Align.center(_logo()))
    console.print(Align.center(Text(_SUBTITLE, style="dim #7df0dd")))
    console.print()
    console.print(_info_panel())


# --------------------------------------------------------------- menu
_MENU = [
    ("1", "Entraîner un LoRA", "train"),
    ("2", "Captionner un dataset", "caption"),
    ("3", "Lancer l'UI web (serve)", "serve"),
    ("4", "Explorer les architectures", "archs"),
    ("q", "Quitter", "quit"),
]


def _menu_panel() -> Panel:
    tb = Table.grid(padding=(0, 2))
    tb.add_column(justify="right"); tb.add_column()
    for key, label, _ in _MENU:
        tb.add_row(f"[bold #43cf9f]{key}[/]", f"[white]{label}[/]")
    return Panel(tb, title="[bold]menu[/]", border_style="#177c72", width=72, padding=(1, 2))


def menu() -> str:
    console.print(Align.center(_menu_panel()))
    choices = [k for k, _, _ in _MENU]
    sel = Prompt.ask("  [bold #43cf9f]>[/] choix", choices=choices, default="1", show_choices=False)
    return dict((k, a) for k, _, a in _MENU)[sel]


# --------------------------------------------------------------- architectures
def browse_archs():
    from families import FAMILIES

    t = Table(title="[bold #7df0dd]Architectures[/]", border_style="#177c72",
              header_style="bold #43cf9f", expand=True)
    t.add_column("id"); t.add_column("modèle"); t.add_column("backend")
    t.add_column("objectif"); t.add_column("quant"); t.add_column("taille", justify="right")
    for f in FAMILIES:
        q = "[#43cf9f]nf4[/]" if f.get("quantizable") else "[dim]bf16[/]"
        t.add_row(f["id"], f["label"], f["backend"], f["prediction"], q, f"~{f.get('params_b','?')}B")
    console.print(t)
    Prompt.ask("  [dim]entrée pour revenir[/]", default="")


# --------------------------------------------------------------- cockpit : helpers
# braille : 2 colonnes × 4 lignes de points par caractère
_BR = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


def _loss_chart(values, w, h) -> Text:
    """Courbe de loss en caractères braille (sparkline haute résolution)."""
    if len(values) < 2:
        return Text("\n" * (h // 2) + "  en attente des premières steps…", style="dim")
    wd, hd = w * 2, h * 4
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    n = len(values)
    grid = [[0] * w for _ in range(h)]
    for xd in range(wd):
        v = values[int(xd / (wd - 1) * (n - 1))]
        yd = int((1 - (v - lo) / rng) * (hd - 1))
        grid[yd // 4][xd // 2] |= _BR[yd % 4][xd % 2]
    t = Text()
    for r in range(h):
        # graduation loss à gauche
        if r == 0:
            t.append(f"{hi:>6.3f} ", style="dim")
        elif r == h - 1:
            t.append(f"{lo:>6.3f} ", style="dim")
        else:
            t.append("       ")
        for c in range(w):
            g = grid[r][c]
            t.append(chr(0x2800 + g) if g else " ", style="#43cf9f")
        t.append("\n")
    return t


def _procedural_preview(sharpness):
    """Aperçu synthétique (mode démo) : un orbe teal qui émerge du bruit."""
    from PIL import Image

    import random as _r

    s = max(0.0, min(1.0, sharpness))
    W = 48
    img = Image.new("RGB", (W, W))
    px = img.load()
    cx = cy = W / 2
    for y in range(W):
        for x in range(W):
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 / (W * 0.55)
            base = max(0.0, 1.0 - d)
            v = base * (0.35 + 0.65 * s) + _r.random() * (1.0 - s) * 0.55
            v = max(0.0, min(1.0, v))
            px[x, y] = (int(25 + 45 * v), int(55 + 195 * v), int(85 + 135 * v))
    return img


def _decode_preview(data_url):
    from PIL import Image

    raw = base64.b64decode(data_url.split(",", 1)[1])
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _img_to_text(img, cols, rows) -> Text:
    """Image → demi-blocs ▀ (2 px verticaux par caractère, truecolor)."""
    from PIL import Image

    img = img.resize((cols, rows * 2), Image.LANCZOS)
    px = img.load()
    t = Text()
    for r in range(rows):
        for c in range(cols):
            tr, tg, tb = px[c, 2 * r]
            br, bg, bb = px[c, 2 * r + 1]
            t.append("▀", style=f"#{tr:02x}{tg:02x}{tb:02x} on #{br:02x}{bg:02x}{bb:02x}")
        t.append("\n")
    return t


_gpu_cache = {"t": 0.0, "d": None}


def _gpu_text() -> Text:
    now = time.time()
    if now - _gpu_cache["t"] > 1.0:
        _gpu_cache["t"] = now
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
                 "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip().splitlines()[0]
            _gpu_cache["d"] = [x.strip() for x in out.split(",")]
        except Exception:
            _gpu_cache["d"] = None
    d = _gpu_cache["d"]
    if not d:
        return Text("  pas de GPU (mode démo)", style="dim")
    util, mu, mt, temp, pw = d

    def bar(frac, w=12):
        f = int(max(0.0, min(1.0, frac)) * w)
        b = Text()
        b.append("█" * f, style="#43cf9f")
        b.append("░" * (w - f), style="dim")
        return b

    t = Text()
    t.append(" charge ", style="dim"); t.append_text(bar(float(util) / 100))
    t.append(f" {util}%\n", style="bold")
    t.append(" vram   ", style="dim"); t.append_text(bar(float(mu) / float(mt)))
    t.append(f" {float(mu)/1024:.1f}/{float(mt)/1024:.0f}G\n", style="bold")
    t.append(f" {temp}°C · {pw}W", style="dim")
    return t


# --------------------------------------------------------------- entraînement (cockpit)
def _train_dashboard(cfg):
    from rich.live import Live

    from trainer import TrainingJob

    losses = deque(maxlen=240)
    logs = deque(maxlen=6)
    preview = {"img": None}
    state = {"done": False, "err": None, "out": ""}

    progress = Progress(
        SpinnerColumn(style="#43cf9f"),
        BarColumn(bar_width=None, complete_style="#43cf9f", finished_style="#7df0dd"),
        TextColumn("[bold]{task.percentage:>3.0f}%"),
        TextColumn("[dim]{task.completed}/{task.total}"),
        TimeElapsedColumn(), TimeRemainingColumn(),
        expand=True,
    )
    task = progress.add_task("", total=cfg.max_steps)

    def emit(e):
        t = e.get("type")
        if t == "log":
            logs.append(("warn" if e.get("level") in ("warn", "error") else "info", e.get("message", "")))
        elif t == "step":
            losses.append(float(e.get("loss", 0.0)))
            progress.update(task, completed=e.get("step", 0), total=e.get("total_steps", cfg.max_steps))
        elif t == "sample":
            try:
                if e.get("placeholder") or not e.get("image"):
                    preview["img"] = _procedural_preview(e.get("sharpness", 0.5))
                else:
                    preview["img"] = _decode_preview(e["image"])
            except Exception:
                pass
        elif t == "status":
            st = e.get("state")
            if st == "done":
                state["done"] = True; state["out"] = e.get("comfyui") or e.get("output") or ""
            elif st == "error":
                state["err"] = e.get("message")

    def _render():
        loss_now = f"{losses[-1]:.4f}" if losses else "—"
        chart = Panel(_loss_chart(list(losses), 40, 8),
                      title=f"[bold]loss[/] [#7df0dd]{loss_now}[/]", border_style="#177c72")
        if preview["img"] is not None:
            prev = Panel(_img_to_text(preview["img"], 34, 16), title="[bold]aperçu du LoRA[/]",
                         border_style="#177c72", padding=0)
        else:
            prev = Panel(Align.center(Text("\n\n\n  l'aperçu apparaîtra ici\n  au 1er échantillon", style="dim"),
                                      vertical="middle"), title="[bold]aperçu du LoRA[/]", border_style="#177c72")
        row = Table.grid(expand=True)
        row.add_column(ratio=3); row.add_column(ratio=2)
        row.add_row(chart, prev)
        log_txt = Text()
        for lvl, msg in logs:
            log_txt.append("  · ", style="#43cf9f" if lvl == "info" else "#e0a54a")
            log_txt.append(msg[:74] + "\n", style="dim")
        bottom = Table.grid(expand=True)
        bottom.add_column(ratio=2); bottom.add_column(ratio=3)
        bottom.add_row(Panel(_gpu_text(), title="[bold]gpu[/]", border_style="#177c72"),
                       Panel(log_txt or Text(" ", style="dim"), title="[bold]journal[/]", border_style="#177c72"))
        return Panel(Group(progress, Text(""), row, bottom),
                     title=f"[bold #7df0dd]{cfg.arch}[/] · {cfg.project_name}",
                     border_style="#43cf9f", padding=(1, 2))

    job = TrainingJob(cfg, emit)
    job.start()
    try:
        with Live(_render(), console=console, refresh_per_second=12, screen=False) as live:
            while job.is_alive():
                live.update(_render()); time.sleep(0.08)
            live.update(_render())
    except KeyboardInterrupt:
        job.stop(); job.join()
        console.print("  [#e0a54a]· arrêt demandé[/]")
    if state["err"]:
        console.print(Panel(f"[bold #e05a5a]✗ erreur[/]  {state['err']}", border_style="#e05a5a"))
    elif state["done"]:
        console.print(Panel(f"[bold #43cf9f]✓ terminé[/]  LoRA → [white]{state['out']}[/]",
                            border_style="#43cf9f"))


def train_flow():
    from config import TrainConfig
    from families import FAMILIES, get_family

    ids = [f["id"] for f in FAMILIES]
    console.print(Align.center(Text("↓ configure ton run (entrée = défaut)", style="dim")))
    # mode d'abord : demo = démonstration sans GPU ni modèle (courbe + aperçu simulés)
    demo = Prompt.ask("  [bold #43cf9f]mode[/]", choices=["demo", "reel"], default="demo") == "demo"
    arch = Prompt.ask("  [bold #43cf9f]archi[/] [dim](sdxl, flux, qwen_image…)[/]",
                      choices=ids, default="sdxl", show_choices=False)
    fam = get_family(arch)
    base = "demo" if demo else Prompt.ask("  [bold #43cf9f]base model[/] (chemin ou repo HF)",
                                          default=fam.get("default_base") or "")
    dataset = "" if demo else Prompt.ask("  [bold #43cf9f]dataset[/] (dossier d'images)")
    steps = int(Prompt.ask("  [bold #43cf9f]steps[/]", default="200" if demo else "1200"))
    cfg = TrainConfig(
        simulate=demo, arch=arch, base_model=base, dataset_dir=dataset,
        project_name=Prompt.ask("  [bold #43cf9f]nom du LoRA[/]", default="my-character"),
        max_steps=steps, resolution=fam.get("resolution", 1024),
        precision="nf4" if fam.get("quantizable") else "bf16",
    )
    cfg.sample_every = max(5, cfg.max_steps // 10)  # aperçus fréquents (voir la précision monter)
    console.print()
    _train_dashboard(cfg)
    Prompt.ask("  [dim]entrée pour revenir au menu[/]", default="")


def caption_flow():
    from captioner import run_captioning
    from config import CaptionConfig

    d = Prompt.ask("  [bold #43cf9f]dataset[/] (dossier d'images)")
    if not d:
        return
    logs = deque(maxlen=1)
    prog = Progress(SpinnerColumn(style="#43cf9f"), TextColumn("{task.description}"),
                    BarColumn(complete_style="#43cf9f"), TextColumn("{task.completed}/{task.total}"))
    task = prog.add_task("captioning", total=1)

    def emit(e):
        if e.get("type") == "caption":
            prog.update(task, total=e.get("total", 1), completed=e.get("index", 0),
                        description=(e.get("file", "") or "")[:30])

    from rich.live import Live

    job = threading.Thread(target=run_captioning,
                           args=(CaptionConfig(dataset_dir=d), emit, threading.Event()), daemon=True)
    job.start()
    with Live(prog, console=console, refresh_per_second=8):
        while job.is_alive():
            time.sleep(0.1)
    console.print("  [bold #43cf9f]✓ captions écrites[/]")
    Prompt.ask("  [dim]entrée pour revenir[/]", default="")


def _port_free(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def serve_flow():
    # défaut = 1er port libre à partir de 8765 (évite le clash avec un moteur déjà lancé)
    default = 8765
    while not _port_free(default) and default < 8790:
        default += 1
    port = int(Prompt.ask("  [bold #43cf9f]port[/] [dim](entrée pour accepter)[/]", default=str(default)))
    if not _port_free(port):
        console.print(f"  [#e0a54a]· le port {port} est déjà utilisé[/] — relance et choisis-en un autre.")
        return
    console.print(Panel(
        f"[bold]UI web[/] → [#7df0dd]http://localhost:{port}/[/]\n"
        f"[dim]Ctrl+C pour arrêter · sur le cloud : http://<ip>:{port}/[/]",
        border_style="#177c72", padding=(0, 2)))
    import uvicorn

    try:
        uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="warning",
                    ws_ping_interval=None, ws_ping_timeout=None)
    except KeyboardInterrupt:
        pass


def run():
    actions = {"train": train_flow, "caption": caption_flow, "serve": serve_flow, "archs": browse_archs}
    first = True
    while True:
        if first:
            intro(); first = False   # intro animée une seule fois
        else:
            splash()
        try:
            choice = menu()
        except (KeyboardInterrupt, EOFError):
            break
        if choice == "quit":
            break
        try:
            actions[choice]()
        except (KeyboardInterrupt, EOFError):
            continue
    console.print("\n  [dim]à bientôt.[/]\n")


if __name__ == "__main__":
    run()
