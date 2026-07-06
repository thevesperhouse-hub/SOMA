"""Create the venv and install the dependencies correctly — including torch cu128
for Blackwell GPUs (RTX 50xx), where standard installs fail.

Usage:
    python bootstrap.py            # base + training (torch cu128)
    python bootstrap.py --base     # server + demo mode only (no torch)
    python bootstrap.py --cuda cu124   # force another CUDA channel

Meant to be run either by hand or by the Tauri app on first launch.
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
    """cu128 for RTX 50xx (Blackwell, sm_120), otherwise cu124."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
        name = out.strip().splitlines()[0] if out.strip() else ""
        print(f"  Detected GPU: {name or 'unknown'}")
        if any(tag in name for tag in ("RTX 50", "5090", "5080", "5070", "5060", "Blackwell")):
            return "cu128"
        return "cu124"
    except Exception:
        print("  nvidia-smi unavailable -> defaulting to cu128 (Blackwell target)")
        return "cu128"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true", help="server + demo only")
    ap.add_argument("--cuda", default=None, help="torch channel (cu128, cu124, cpu)")
    args = ap.parse_args()

    print(">> SOMA bootstrap")
    if not os.path.isdir(VENV_DIR):
        print(f">> Creating the venv: {VENV_DIR}")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    else:
        print(">> venv already present")

    py = venv_python()
    run([py, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])

    print(">> Base dependencies (server + demo)")
    run([py, "-m", "pip", "install", "-r", os.path.join(HERE, "requirements-base.txt")])

    if args.base:
        print(">> OK (base mode). Demo available: python server.py")
        return

    channel = args.cuda or detect_cuda_channel()
    print(f">> torch / torchvision via channel {channel}")
    if channel == "cpu":
        run([py, "-m", "pip", "install", "torch", "torchvision"])
    else:
        run([py, "-m", "pip", "install", "torch", "torchvision",
             "--index-url", f"https://download.pytorch.org/whl/{channel}"])

    print(">> Training dependencies (diffusers + peft …)")
    run([py, "-m", "pip", "install", "-r", os.path.join(HERE, "requirements-train.txt")])

    print(">> Checking torch + CUDA")
    try:
        run([py, "-c",
             "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),"
             "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"])
    except subprocess.CalledProcessError:
        print("!! torch imported but CUDA check failed — see the message above")

    print(">> Done. Start the engine: python server.py")


if __name__ == "__main__":
    main()
