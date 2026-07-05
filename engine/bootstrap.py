"""Crée le venv et installe les dépendances correctement — y compris torch cu128
pour les GPU Blackwell (RTX 50xx), là où les installs standard plantent.

Usage :
    python bootstrap.py            # base + entraînement (torch cu128)
    python bootstrap.py --base     # serveur + mode démo seulement (pas de torch)
    python bootstrap.py --cuda cu124   # forcer un autre channel CUDA

Pensé pour être appelé soit à la main, soit par l'app Tauri au 1er lancement.
"""
import argparse
import os
import subprocess
import sys
import venv

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(HERE, ".venv")
IS_WIN = os.name == "nt"


def venv_python(d=VENV_DIR):
    return os.path.join(d, "Scripts" if IS_WIN else "bin", "python.exe" if IS_WIN else "python")


def run(cmd):
    print("  $", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def detect_cuda_channel():
    """cu128 pour les RTX 50xx (Blackwell, sm_120), sinon cu124."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
        name = out.strip().splitlines()[0] if out.strip() else ""
        print(f"  GPU détecté : {name or 'inconnu'}")
        if any(tag in name for tag in ("RTX 50", "5090", "5080", "5070", "5060", "Blackwell")):
            return "cu128"
        return "cu124"
    except Exception:
        print("  nvidia-smi indisponible -> défaut cu128 (cible Blackwell)")
        return "cu128"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true", help="serveur + démo seulement")
    ap.add_argument("--cuda", default=None, help="channel torch (cu128, cu124, cpu)")
    args = ap.parse_args()

    print(">> SOMA bootstrap")
    if not os.path.isdir(VENV_DIR):
        print(f">> Création du venv : {VENV_DIR}")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    else:
        print(">> venv déjà présent")

    py = venv_python()
    run([py, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])

    print(">> Dépendances de base (serveur + démo)")
    run([py, "-m", "pip", "install", "-r", os.path.join(HERE, "requirements-base.txt")])

    if args.base:
        print(">> OK (mode base). Démo dispo : python server.py")
        return

    channel = args.cuda or detect_cuda_channel()
    print(f">> torch / torchvision via channel {channel}")
    if channel == "cpu":
        run([py, "-m", "pip", "install", "torch", "torchvision"])
    else:
        run([py, "-m", "pip", "install", "torch", "torchvision",
             "--index-url", f"https://download.pytorch.org/whl/{channel}"])

    print(">> Dépendances d'entraînement (diffusers + peft …)")
    run([py, "-m", "pip", "install", "-r", os.path.join(HERE, "requirements-train.txt")])

    print(">> Vérification torch + CUDA")
    try:
        run([py, "-c",
             "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),"
             "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"])
    except subprocess.CalledProcessError:
        print("!! torch importé mais vérif CUDA en échec — voir le message ci-dessus")

    print(">> Terminé. Lancer le moteur : python server.py")


if __name__ == "__main__":
    main()
