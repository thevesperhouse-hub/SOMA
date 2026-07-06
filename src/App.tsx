import { useEffect, useMemo, useRef, useState } from "react";
import { connectEvents, startCaptioning, startTraining, stopCaptioning, stopTraining } from "./lib/api";
import { computeXp, unlockedAchievements, type GameStats } from "./lib/game";
import { getInitialLang, makeT, type Lang } from "./lib/i18n";
import { applyTheme, getInitialTheme, type Theme } from "./lib/theme";
import { cn, fmtLr, fmtSecs } from "./lib/utils";
import type { CaptionConfig, Sample, TrainConfig, TrainEvent, TrainState } from "./types";
import { Card, CardHeader, Stat, Progress, Badge } from "./components/ui";
import { LossChart } from "./components/LossChart";
import { SampleViewer } from "./components/SampleViewer";
import { GamePanel } from "./components/GamePanel";
import { ConfigPanel } from "./components/ConfigPanel";
import { DatasetView } from "./components/DatasetView";
import { GpuMonitor } from "./components/GpuMonitor";
import { ModelsView } from "./components/ModelsView";
import { SettingsView } from "./components/SettingsView";
import { Sidebar, type View } from "./components/Sidebar";

const DEFAULT_CFG: TrainConfig = {
  project_name: "my-character",
  arch: "sdxl",
  base_model: "stabilityai/stable-diffusion-xl-base-1.0",
  dataset_dir: "",
  instance_token: "ohwx",
  output_dir: "output",
  resolution: 1024,
  rank: 16,
  alpha: 8,
  learning_rate: 0.0001,
  max_steps: 1200,
  batch_size: 1,
  lr_warmup_ratio: 0.05,
  min_snr_gamma: 5.0,
  caption_dropout: 0.1,
  gradient_checkpointing: true,
  mixed_precision: "bf16",
  precision: "bf16",
  sample_every: 100,
  sample_prompt: "a portrait photo of ohwx person, natural light, sharp focus",
  seed: 42,
  simulate: true,
};

interface RunState {
  losses: number[];
  step: number;
  total: number;
  lr: number;
  secs: number;
  samples: Sample[];
  first: number;
  best: number;
  finished: boolean;
}

const EMPTY_RUN: RunState = {
  losses: [], step: 0, total: 0, lr: 0, secs: 0, samples: [], first: 0, best: Infinity, finished: false,
};

export default function App() {
  const [cfg, setCfg] = useState<TrainConfig>(DEFAULT_CFG);
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState<TrainState>("idle");
  const [run, setRun] = useState<RunState>(EMPTY_RUN);
  const [gameOn, setGameOn] = useState(true);
  const [lang, setLang] = useState<Lang>(getInitialLang);
  const t = useMemo(() => makeT(lang), [lang]);
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  useEffect(() => { applyTheme(theme); }, [theme]);
  const [view, setView] = useState<View>(() => {
    const saved = localStorage.getItem("soma.view");
    // migrate old values
    if (saved === "train") return "dashboard";
    if (saved === "dataset") return "datasets";
    const valid: View[] = ["dashboard", "datasets", "models", "settings"];
    return valid.includes(saved as View) ? (saved as View) : "dashboard";
  });
  const [caption, setCaption] = useState<{
    running: boolean; index: number; total: number; current: string; map: Record<string, string>;
  }>({ running: false, index: 0, total: 0, current: "", map: {} });
  // JoyCaption model download/load state (shown as a badge + progress in DatasetView)
  const [captionModel, setCaptionModel] = useState<{
    state: "idle" | "downloading" | "loading" | "ready"; percent: number;
  }>({ state: "idle", percent: 0 });
  const [note, setNote] = useState(""); // latest engine log (loading, cache…)
  const [ckptRefresh, setCkptRefresh] = useState(0); // bump -> reload the LoRA list

  // Persistent GLOBAL XP (accumulated across all LoRAs) — the current run's XP is
  // added live, and banked permanently when the run finishes.
  const [committedXp, setCommittedXp] = useState<number>(
    () => Number(localStorage.getItem("soma.xp.global") || 0)
  );
  const bankedRef = useRef(false);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let offline: ReturnType<typeof setTimeout> | undefined;
    let closed = false;
    const open = () => {
      ws = connectEvents(
        handleEvent,
        () => { clearTimeout(offline); if (!closed) setConnected(true); },
        () => {
          // Effect cleaned up (StrictMode dev / unmount) -> ignore entirely, otherwise
          // this onclose (which runs AFTER the cleanup) would reschedule a phantom
          // setConnected(false) and grey out the button on its own.
          if (closed) return;
          // The WS briefly disconnects on EVERY image (the server loop is starved by
          // the model). We do NOT flap `connected`: it stays "online" across these
          // micro-reconnects (otherwise the whole app re-renders on every image =
          // blink). Offline only after a real 4s.
          clearTimeout(offline);
          offline = setTimeout(() => { if (!closed) setConnected(false); }, 4000);
          retry = setTimeout(open, 600); // auto-reconnect
        }
      );
    };
    open();
    const ping = setInterval(() => ws?.readyState === 1 && ws.send("ping"), 15000);
    return () => {
      closed = true;
      clearInterval(ping);
      clearTimeout(retry);
      clearTimeout(offline);
      ws?.close();
    };
  }, []);

  useEffect(() => { localStorage.setItem("soma.view", view); }, [view]);
  useEffect(() => { localStorage.setItem("soma.lang", lang); }, [lang]);

  function handleEvent(e: TrainEvent) {
    if (e.type === "log") {
      if (e.level !== "error") setNote(e.message); // shows the loading progress
      return;
    }
    if (e.type === "caption_model") {
      setCaptionModel({ state: e.state, percent: e.percent ?? 0 });
      return;
    }
    if (e.type === "status") {
      setState(e.state);
      if (e.state === "starting") { setRun(EMPTY_RUN); setNote(t("init")); }
      if (e.state === "done") { setRun((r) => ({ ...r, finished: true })); setCkptRefresh((n) => n + 1); }
      if (e.state === "captioning")
        setCaption((c) => ({ ...c, running: true, total: (e as { total?: number }).total ?? c.total }));
      if (e.state === "done_caption" || e.state === "stopped" || e.state === "error")
        setCaption((c) => ({ ...c, running: false, current: "" }));
    } else if (e.type === "caption") {
      setCaption((c) => ({
        ...c,
        running: true,
        index: e.index,
        total: e.total,
        current: e.file,
        map: e.skipped ? c.map : { ...c.map, [e.file]: e.text },
      }));
    } else if (e.type === "step") {
      setRun((r) => ({
        ...r,
        losses: [...r.losses, e.loss],
        step: e.step,
        total: e.total_steps,
        lr: e.lr,
        secs: e.secs,
        first: r.first || e.loss,
        best: Math.min(r.best, e.loss),
      }));
    } else if (e.type === "sample") {
      setRun((r) => ({
        ...r,
        samples: [
          ...r.samples,
          {
            step: e.step, total: e.total_steps, placeholder: e.placeholder,
            image: e.image, seed: e.seed ?? 0, sharpness: e.sharpness,
          },
        ],
      }));
    }
  }

  const stats: GameStats = {
    steps: run.step,
    totalSteps: run.total,
    bestLoss: run.best === Infinity ? 0 : run.best,
    lastLoss: run.losses[run.losses.length - 1] ?? 0,
    firstLoss: run.first,
    samples: run.samples.length,
    finished: run.finished,
  };
  const unlocked = useMemo(() => new Set(unlockedAchievements(stats)), [
    stats.steps, stats.bestLoss, stats.samples, stats.finished,
  ]);
  const xp = computeXp(stats, unlocked.size);

  // Displayed global XP = bank + current run's XP (rising live); when the run ends,
  // we bank once (bankedRef) -> continuity without double-counting.
  useEffect(() => {
    if (run.finished && !bankedRef.current) {
      bankedRef.current = true;
      setCommittedXp((c) => {
        const n = c + xp;
        localStorage.setItem("soma.xp.global", String(n));
        return n;
      });
    } else if (!run.finished) {
      bankedRef.current = false;
    }
  }, [run.finished, xp]);
  const globalXp = committedXp + (run.finished ? 0 : xp);

  const progress = run.total ? run.step / run.total : 0;
  const perStep = run.step ? run.secs / run.step : 0;
  const eta = run.total && run.step ? (run.total - run.step) * perStep : NaN;
  const running = state === "training" || state === "sampling" || state === "starting";

  async function onStart() {
    setRun(EMPTY_RUN);
    const r = await startTraining(cfg);
    if (!r.ok) alert(r.error ?? t("err.startFail"));
  }

  async function onCaptionStart(c: CaptionConfig) {
    const r = await startCaptioning(c);
    if (!r.ok) alert(r.error ?? t("err.captionFail"));
  }

  return (
    <div className="flex h-screen">
      <Sidebar view={view} setView={setView} connected={connected} t={t} />

      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Top bar : titre de section + statut du run */}
        <header className="soma-header flex h-14 shrink-0 items-center justify-between border-b border-border px-6">
          <h1 className="text-sm font-semibold">{t("nav." + view)}</h1>
          <div className="flex items-center gap-3">
            {running &&
              (run.step === 0 ? (
                <span className="max-w-[280px] truncate text-xs text-muted">{note || t("loading")}</span>
              ) : (
                <span className="text-xs text-muted tabular-nums">ETA {fmtSecs(eta)}</span>
              ))}
            <Badge tone={state === "error" ? "bad" : state === "done" ? "good" : running ? "accent" : "muted"}>
              {t("state." + state)}
            </Badge>
          </div>
        </header>

        <main className="flex flex-1 flex-col overflow-hidden">
          {/* Dashboard: training (3 columns) */}
          {view === "dashboard" && (
          <div className="soma-grid grid h-full grid-cols-[340px_1fr_320px] gap-4 overflow-hidden p-4">
        {/* Gauche : config */}
        <div className="soma-col-a overflow-hidden">
          <ConfigPanel
            cfg={cfg}
            setCfg={setCfg}
            state={state}
            connected={connected}
            onStart={onStart}
            onStop={stopTraining}
            t={t}
          />
        </div>

        {/* Centre : dashboard */}
        <div className="soma-col-b flex flex-col gap-4 overflow-y-auto">
          <div className="grid grid-cols-5 gap-3">
            <Stat label={t("dash.step")} value={`${run.step}`} sub={run.total ? `/ ${run.total}` : undefined} />
            <Stat label={t("dash.loss")} value={stats.lastLoss ? stats.lastLoss.toFixed(4) : "—"} sub={run.best !== Infinity ? `${t("dash.best")} ${run.best.toFixed(4)}` : undefined} />
            <Stat label={t("dash.speed")} value={perStep ? `${perStep.toFixed(2)}s` : "—"} sub="/ it" />
            <Stat label="LR" value={fmtLr(run.lr)} />
            <Stat label={t("dash.elapsed")} value={fmtSecs(run.secs)} />
          </div>

          <GpuMonitor connected={connected} t={t} />

          <Card>
            <CardHeader
              title={t("dash.progress")}
              hint={run.total ? `${Math.round(progress * 100)}%` : t("dash.prep")}
            />
            <div className="px-5 pb-4">
              <Progress value={progress} />
              {note && run.step === 0 && running && (
                <div className="mt-2 flex items-center gap-2 text-xs text-muted">
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
                  <span className="truncate">{note}</span>
                </div>
              )}
            </div>
          </Card>

          <div className="grid grid-cols-[1fr_320px] gap-4">
            <Card>
              <CardHeader title={t("dash.lossCurve")} hint={t("dash.lossHint")} />
              <LossChart losses={run.losses} />
            </Card>
            <Card>
              <CardHeader title={t("dash.preview")} hint={t("dash.previewHint")} />
              <SampleViewer samples={run.samples} />
            </Card>
          </div>
        </div>

        {/* Droite : gamification */}
        <div className="soma-col-c overflow-hidden">
          <GamePanel
            runXp={xp}
            globalXp={globalXp}
            unlocked={unlocked}
            enabled={gameOn}
            onToggle={() => setGameOn(!gameOn)}
            t={t}
          />
        </div>
      </div>
          )}

          {view === "datasets" && (
            <DatasetView
              caption={caption}
              captionModel={captionModel}
              connected={connected}
              onStart={onCaptionStart}
              onStop={stopCaptioning}
              onReset={() => setCaption({ running: false, index: 0, total: 0, current: "", map: {} })}
              t={t}
            />
          )}

          {view === "models" && (
            <ModelsView connected={connected} outputDir={cfg.output_dir} refreshKey={ckptRefresh} t={t} />
          )}

          {view === "settings" && (
            <SettingsView
              theme={theme}
              setTheme={setTheme}
              lang={lang}
              setLang={setLang}
              gameOn={gameOn}
              setGameOn={setGameOn}
              t={t}
            />
          )}
        </main>
      </div>
    </div>
  );
}
