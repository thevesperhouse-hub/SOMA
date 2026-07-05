import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Port 1420 = défaut attendu par Tauri. Tailwind v4 via son plugin Vite.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    // ⚠️ CRUCIAL : sans ça, Vite surveille tout le projet et RECHARGE la page à
    // chaque fichier écrit hors de src/ — les .txt de captions, les logs du
    // moteur, les .safetensors des LoRA… -> "toute la page blink" + perte d'état
    // (le token qui s'efface) à chaque image captionnée. On ignore tout ce qui
    // n'est pas du code source.
    watch: {
      ignored: [
        "**/engine/**", // moteur Python : .venv, output, cache, logs
        "**/*.txt", // captions écrites pendant le captioning
        "**/*.safetensors", // LoRA écrits pendant l'entraînement
        "**/lora01/**", // dataset de test à la racine du projet
        "**/dist/**",
      ],
    },
  },
});
