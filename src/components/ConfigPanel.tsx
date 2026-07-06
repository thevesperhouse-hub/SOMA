import { useEffect, useState } from "react";
import type { TrainConfig, TrainState } from "../types";
import { datasetList, gpuInfo, listFamilies, listModels, type Family, type ModelEntry } from "../lib/api";
import type { T } from "../lib/i18n";
import { Button, Card, CardHeader, Field, Input } from "./ui";
import { cn } from "../lib/utils";

export function ConfigPanel({
  cfg,
  setCfg,
  state,
  connected,
  onStart,
  onStop,
  t,
}: {
  cfg: TrainConfig;
  setCfg: (c: TrainConfig) => void;
  state: TrainState;
  connected: boolean;
  onStart: () => void;
  onStop: () => void;
  t: T;
}) {
  const running = state === "training" || state === "sampling" || state === "starting";
  const set = <K extends keyof TrainConfig>(k: K, v: TrainConfig[K]) => setCfg({ ...cfg, [k]: v });

  // Model families (engine-side registry): populate the selector + the defaults.
  const [families, setFamilies] = useState<Family[]>([]);
  useEffect(() => {
    if (!connected) return;
    let cancel = false;
    listFamilies().then((f) => !cancel && f.length && setFamilies(f));
    return () => { cancel = true; };
  }, [connected]);
  const fam = families.find((f) => f.id === cfg.arch);

  // Detected VRAM -> smart precision default (never forced, overridable).
  const [vram, setVram] = useState(0);
  useEffect(() => {
    if (!connected) return;
    let cancel = false;
    gpuInfo().then((g) => !cancel && g.cuda && setVram(g.vram_gb));
    return () => { cancel = true; };
  }, [connected]);

  // bf16 if the weights + activation headroom fit; otherwise nf4 (if quantizable).
  const recommendPrecision = (f: Family | undefined, v: number): string => {
    if (!f || !f.quantizable || v <= 0) return "bf16";
    const bf16Weights = f.params_b * 2; // Go
    return bf16Weights + 5 <= v ? "bf16" : "nf4";
  };

  // Family change: switches the defaults (base, resolution, checkpointing, precision)
  // only if the user hadn't set a custom value.
  const setArch = (arch: string) => {
    const next = families.find((f) => f.id === arch);
    if (!next) { set("arch", arch); return; }
    const defaultBases = families.map((f) => f.default_base).filter(Boolean);
    const wasDefaultBase = defaultBases.includes(cfg.base_model);
    const wasDefaultRes = families.some((f) => f.resolution === cfg.resolution);
    const res = wasDefaultRes ? next.resolution : cfg.resolution;
    setCfg({
      ...cfg,
      arch,
      base_model: wasDefaultBase ? next.default_base : cfg.base_model,
      resolution: res,
      gradient_checkpointing: res > 768, // ≤768: OFF (speed); 1024: ON (else OOM)
      precision: recommendPrecision(next, vram),
    });
  };

  // Models folder (ComfyUI) + arch-filtered checkpoint list
  const [modelRoot, setModelRoot] = useState(() => localStorage.getItem("soma.modelRoot") || "");
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);

  const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
  async function pickModelRoot() {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const sel = await open({ directory: true, multiple: false, title: t("picker.modelsDir") });
      if (typeof sel === "string") setModelRoot(sel);
    } catch {
      /* browser: no native picker */
    }
  }

  useEffect(() => { localStorage.setItem("soma.modelRoot", modelRoot); }, [modelRoot]);

  // (re)load the list when the arch, folder or connection changes
  useEffect(() => {
    if (!connected) return;
    let cancel = false;
    setLoadingModels(true);
    listModels(cfg.arch, modelRoot)
      .then((r) => {
        if (cancel) return;
        setModels(r.models);
        if (!modelRoot && r.root) setModelRoot(r.root); // adopt the auto-detected folder
      })
      .finally(() => !cancel && setLoadingModels(false));
    return () => { cancel = true; };
  }, [cfg.arch, modelRoot, connected]);

  const known = models.some((m) => m.path === cfg.base_model);

  // Dataset: selection + auto-detection of captions (.txt next to the images).
  // Synced with the Dataset tab via localStorage "soma.dir".
  const cleanDir = (d: string) => d.trim().replace(/^["']+|["']+$/g, "").trim();
  const [ds, setDs] = useState<{ total: number; captioned: number } | null>(null);
  const [dsLoading, setDsLoading] = useState(false);

  async function pickDataset() {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const sel = await open({ directory: true, multiple: false, title: t("picker.dataset") });
      if (typeof sel === "string") set("dataset_dir", sel);
    } catch {
      /* browser: no native picker */
    }
  }

  // on mount: if no dataset, reuse the one from the Dataset tab
  useEffect(() => {
    if (!cfg.dataset_dir) {
      const saved = localStorage.getItem("soma.dir");
      if (saved) set("dataset_dir", saved);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // detect images + captions as soon as a folder is set
  useEffect(() => {
    const dir = cleanDir(cfg.dataset_dir);
    if (dir) localStorage.setItem("soma.dir", cfg.dataset_dir);
    if (!connected || !dir) { setDs(null); return; }
    let cancel = false;
    setDsLoading(true);
    datasetList(dir)
      .then((imgs) => {
        if (cancel) return;
        setDs({ total: imgs.length, captioned: imgs.filter((i) => i.caption.trim()).length });
      })
      .finally(() => !cancel && setDsLoading(false));
    return () => { cancel = true; };
  }, [cfg.dataset_dir, connected]);

  return (
    <Card className="flex h-full flex-col">
      <CardHeader
        title={t("cfg.title")}
        hint={fam?.backend === "zimage" ? t("cfg.hint.zimage") : t("cfg.hint.sdxl")}
        right={
          <span
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs",
              connected ? "bg-good/15 text-good" : "bg-bad/15 text-bad"
            )}
          >
            <span className={cn("h-1.5 w-1.5 rounded-full", connected ? "bg-good" : "bg-bad")} />
            {connected ? t("cfg.engineOk") : t("cfg.offline")}
          </span>
        }
      />

      <div className="flex flex-1 flex-col gap-3 overflow-y-auto px-5 pb-3">
        <Field label={t("cfg.project")}>
          <Input value={cfg.project_name} onChange={(e) => set("project_name", e.target.value)} />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label={t("cfg.arch")}>
            <select
              value={cfg.arch}
              onChange={(e) => setArch(e.target.value)}
              className="w-full rounded-lg border border-border bg-surface-2 px-3 py-2 text-sm outline-none focus:border-accent"
            >
              {(families.length ? families : [{ id: cfg.arch, label: cfg.arch }]).map((f) => (
                <option key={f.id} value={f.id}>
                  {f.label}
                </option>
              ))}
            </select>
          </Field>
          <Field label={t("cfg.resolution")}>
            <Input
              type="number"
              step="64"
              value={cfg.resolution}
              onChange={(e) => set("resolution", Number(e.target.value))}
            />
          </Field>
        </div>
        <Field
          label={t("cfg.precision")}
          hint={
            fam && vram > 0
              ? `${vram} GB VRAM · ~${(fam.params_b * (cfg.precision === "nf4" ? 0.5 : cfg.precision === "int8" ? 1 : 2)).toFixed(1)} GB weights`
              : t("cfg.precisionHint")
          }
        >
          <select
            value={cfg.precision}
            onChange={(e) => set("precision", e.target.value)}
            disabled={!fam?.quantizable}
            className="w-full rounded-lg border border-border bg-surface-2 px-3 py-2 text-sm outline-none focus:border-accent disabled:opacity-50"
          >
            <option value="bf16">bf16 — no quantization</option>
            <option value="int8">int8 — 8-bit</option>
            <option value="nf4">nf4 — 4-bit (min VRAM)</option>
          </select>
        </Field>
        {fam?.quantizable && vram > 0 && recommendPrecision(fam, vram) !== cfg.precision && (
          <div
            className={cn(
              "rounded-lg px-3 py-2 text-xs",
              recommendPrecision(fam, vram) === "nf4" ? "bg-warn/15 text-warn" : "bg-accent/15 text-accent"
            )}
          >
            {recommendPrecision(fam, vram) === "nf4"
              ? `${fam.label} in bf16 (~${(fam.params_b * 2).toFixed(0)} GB of weights) may overflow your ${vram} GB — nf4 recommended.`
              : `Your ${vram} GB allow bf16 (better quality).`}
          </div>
        )}
        <Field label={t("cfg.modelsDir")} hint={t("cfg.modelsDirHint")}>
          <div className="flex gap-2">
            <Input
              value={modelRoot}
              placeholder="…/ComfyUI/models"
              onChange={(e) => setModelRoot(e.target.value)}
            />
            {inTauri && (
              <Button variant="ghost" onClick={pickModelRoot}>{t("cfg.browse")}</Button>
            )}
          </div>
        </Field>
        <Field
          label={t("cfg.model")}
          hint={
            loadingModels
              ? t("cfg.searching")
              : `${models.length} ${t("cfg.found")} · ${fam?.backend === "zimage" ? t("cfg.foundZ") : t("cfg.foundS")}`
          }
        >
          <select
            value={known ? cfg.base_model : "__custom__"}
            onChange={(e) =>
              e.target.value !== "__custom__" && set("base_model", e.target.value)
            }
            className="w-full rounded-lg border border-border bg-surface-2 px-3 py-2 text-sm outline-none focus:border-accent"
          >
            {models.map((m) => (
              <option key={m.path} value={m.path}>
                {m.name}
                {m.folder === "diffusion_models" ? "  (DiT)" : ""}
              </option>
            ))}
            <option value="__custom__">{t("cfg.custom")}</option>
          </select>
        </Field>
        {!known && (
          <Field
            label={t("cfg.customPath")}
            hint={fam?.backend === "zimage" ? t("cfg.customHintZ") : t("cfg.customHintS")}
          >
            <Input
              value={cfg.base_model}
              placeholder={fam?.default_base || "checkpoint .safetensors / repo HF"}
              onChange={(e) => set("base_model", e.target.value)}
            />
          </Field>
        )}
        <Field label={t("cfg.dataset")} hint={t("cfg.datasetHint")}>
          <div className="flex gap-2">
            <Input
              value={cfg.dataset_dir}
              placeholder="C:\\...\\dataset"
              onChange={(e) => set("dataset_dir", e.target.value)}
            />
            {inTauri && (
              <Button variant="ghost" onClick={pickDataset}>{t("cfg.browse")}</Button>
            )}
          </div>
          {cleanDir(cfg.dataset_dir) && (
            <div className="mt-1.5 flex items-center gap-2 text-xs">
              {dsLoading ? (
                <span className="text-muted">{t("cfg.dsAnalyzing")}</span>
              ) : ds ? (
                <>
                  <span className="text-muted">{ds.total} {t("cfg.dsImages")}</span>
                  {ds.total > 0 && (
                    <span
                      className={cn(
                        "inline-flex items-center gap-1 rounded-full px-2 py-0.5",
                        ds.captioned === 0
                          ? "bg-border/60 text-muted"
                          : ds.captioned === ds.total
                          ? "bg-good/15 text-good"
                          : "bg-accent/15 text-accent"
                      )}
                    >
                      {ds.captioned === 0
                        ? t("cfg.dsNoCap")
                        : ds.captioned === ds.total
                        ? t("cfg.dsAllCap")
                        : `${ds.captioned}/${ds.total} ${t("cfg.dsSomeCap")}`}
                    </span>
                  )}
                </>
              ) : (
                <span className="text-bad">{t("cfg.dsNotFound")}</span>
              )}
            </div>
          )}
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label={t("cfg.token")}>
            <Input value={cfg.instance_token} onChange={(e) => set("instance_token", e.target.value)} />
          </Field>
          <Field label={t("cfg.steps")}>
            <Input
              type="number"
              value={cfg.max_steps}
              onChange={(e) => set("max_steps", Number(e.target.value))}
            />
          </Field>
        </div>
        <div className="grid grid-cols-3 gap-3">
          <Field label={t("cfg.rank")}>
            <Input type="number" value={cfg.rank} onChange={(e) => set("rank", Number(e.target.value))} />
          </Field>
          <Field label={t("cfg.alpha")}>
            <Input type="number" value={cfg.alpha} onChange={(e) => set("alpha", Number(e.target.value))} />
          </Field>
          <Field label={t("cfg.lr")}>
            <Input
              type="number"
              step="0.00001"
              value={cfg.learning_rate}
              onChange={(e) => set("learning_rate", Number(e.target.value))}
            />
          </Field>
        </div>
        <Field
          label={t("cfg.samplePrompt")}
          hint={fam?.prompt_hint ? fam.prompt_hint.replace(/<token>/g, cfg.instance_token || "ohwx") : undefined}
        >
          <Input value={cfg.sample_prompt} onChange={(e) => set("sample_prompt", e.target.value)} />
        </Field>

        <label className="mt-1 flex items-center justify-between rounded-xl border border-border bg-surface-2 px-3 py-2.5">
          <div>
            <div className="text-sm font-medium">{t("cfg.gradCkpt")}</div>
            <div className="text-xs text-muted">
              {cfg.resolution > 768 ? t("cfg.gradCkptOn") : t("cfg.gradCkptOff")}
            </div>
          </div>
          <button
            onClick={() => set("gradient_checkpointing", !cfg.gradient_checkpointing)}
            className={cn(
              "relative h-6 w-11 rounded-full transition",
              cfg.gradient_checkpointing ? "bg-accent" : "bg-border"
            )}
          >
            <span
              className={cn(
                "absolute top-0.5 h-5 w-5 rounded-full bg-white transition-all",
                cfg.gradient_checkpointing ? "left-[22px]" : "left-0.5"
              )}
            />
          </button>
        </label>
        {fam?.backend === "zimage" && cfg.resolution >= 1024 && !cfg.gradient_checkpointing && (
          <div className="rounded-lg bg-bad/15 px-3 py-2 text-xs text-bad">{t("cfg.oomWarn")}</div>
        )}

        <label className="mt-1 flex items-center justify-between rounded-xl border border-border bg-surface-2 px-3 py-2.5">
          <div>
            <div className="text-sm font-medium">{t("cfg.demo")}</div>
            <div className="text-xs text-muted">{t("cfg.demoSub")}</div>
          </div>
          <button
            onClick={() => set("simulate", !cfg.simulate)}
            className={cn(
              "relative h-6 w-11 rounded-full transition",
              cfg.simulate ? "bg-accent" : "bg-border"
            )}
          >
            <span
              className={cn(
                "absolute top-0.5 h-5 w-5 rounded-full bg-white transition-all",
                cfg.simulate ? "left-[22px]" : "left-0.5"
              )}
            />
          </button>
        </label>
      </div>

      <div className="flex gap-2 border-t border-border p-4">
        {running ? (
          <Button variant="danger" className="flex-1" onClick={onStop}>
            {t("cfg.stop")}
          </Button>
        ) : (
          <Button variant="primary" className="flex-1" disabled={!connected} onClick={onStart}>
            {t("cfg.start")}
          </Button>
        )}
      </div>
    </Card>
  );
}
