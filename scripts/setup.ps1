# SOMA — installation en une commande (Windows).
# Crée le venv Python (avec torch cu128 pour Blackwell), installe le front.
# Usage :  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#          scripts\setup.ps1 -Base    # serveur + démo seulement (pas de torch)

param([switch]$Base)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==> SOMA setup" -ForegroundColor Cyan

# 1) Moteur Python + venv + deps
Write-Host "==> Python engine (venv + deps)" -ForegroundColor Cyan
if ($Base) {
    python "engine\bootstrap.py" --base
} else {
    python "engine\bootstrap.py"
}

# 2) Frontend
Write-Host "==> Frontend (npm install)" -ForegroundColor Cyan
npm install

Write-Host ""
Write-Host "OK. Pour lancer :" -ForegroundColor Green
Write-Host "   npm run tauri dev      (l'app desktop ; lance le moteur toute seule)"
Write-Host "ou, pour tester vite le dashboard seul :"
Write-Host "   1) engine\.venv\Scripts\python engine\server.py"
Write-Host "   2) npm run dev   puis ouvrir http://localhost:1420"
