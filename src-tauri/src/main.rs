// Empêche l'ouverture d'une console Windows en release.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;

use tauri::Manager;

/// Garde le process du moteur Python pour le tuer à la fermeture.
struct Engine(Mutex<Option<Child>>);

/// Lance `engine/.venv/python server.py` à la racine du projet.
fn spawn_engine() -> Option<Child> {
    let cwd = std::env::current_dir().ok()?;
    // En dev, current_dir = src-tauri ; la racine projet est le parent.
    let root: PathBuf = cwd.parent().map(|p| p.to_path_buf()).unwrap_or(cwd.clone());
    let engine_dir = root.join("engine");

    let venv_py = if cfg!(windows) {
        engine_dir.join(".venv").join("Scripts").join("python.exe")
    } else {
        engine_dir.join(".venv").join("bin").join("python")
    };
    let python = if venv_py.exists() {
        venv_py
    } else {
        PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
    };

    let mut cmd = Command::new(python);
    cmd.arg(engine_dir.join("server.py"));
    cmd.current_dir(&engine_dir);

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    match cmd.spawn() {
        Ok(child) => Some(child),
        Err(e) => {
            eprintln!("[SOMA] impossible de lancer le moteur Python: {e}");
            None
        }
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            app.manage(Engine(Mutex::new(spawn_engine())));
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(engine) = window.app_handle().try_state::<Engine>() {
                    if let Ok(mut guard) = engine.0.lock() {
                        if let Some(mut child) = guard.take() {
                            let _ = child.kill();
                        }
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("erreur au lancement de SOMA");
}
