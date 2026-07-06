import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CaptionConfig, DatasetImage } from "../types";
import { captionModelStatus, datasetList, datasetThumbUrl, saveCaption, uploadDataset } from "../lib/api";
import type { T } from "../lib/i18n";
import { cn } from "../lib/utils";
import { Button, Card, CardHeader, Field, Input, Progress } from "./ui";

export interface LiveCaption {
  running: boolean;
  index: number;
  total: number;
  current: string;
  map: Record<string, string>;
}

const DEFAULT_PROMPT = "Write a detailed description for this image.";

export interface CaptionModel {
  state: "idle" | "downloading" | "loading" | "ready";
  percent: number;
}

export function DatasetView({
  caption,
  captionModel,
  connected,
  onStart,
  onStop,
  onReset,
  t,
}: {
  caption: LiveCaption;
  captionModel: CaptionModel;
  connected: boolean;
  onStart: (cfg: CaptionConfig) => void;
  onStop: () => void;
  onReset: () => void;
  t: T;
}) {
  // Is JoyCaption already downloaded? (null = unknown / checking)
  const [modelCached, setModelCached] = useState<boolean | null>(null);
  useEffect(() => {
    let alive = true;
    captionModelStatus().then((s) => { if (alive) setModelCached(s.cached); });
    return () => { alive = false; };
  }, []);
  // Once a download/load completed, treat the model as available.
  useEffect(() => {
    if (captionModel.state === "ready") setModelCached(true);
  }, [captionModel.state]);
  const [dir, setDir] = useState(() => localStorage.getItem("soma.dir") || "");
  const [token, setToken] = useState("ohwx");
  const [overwrite, setOverwrite] = useState(true);
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
  async function pickFolder() {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const sel = await open({ directory: true, multiple: false, title: t("picker.dataset") });
      if (typeof sel === "string") setDir(sel);
    } catch {
      /* browser: no native picker */
    }
  }

  const cleanDir = () => dir.trim().replace(/^["']+|["']+$/g, "").trim();

  async function onUpload(files: FileList | null) {
    const d = cleanDir();
    if (!files || !files.length || !d) return;
    setUploading(true);
    try {
      await uploadDataset(d, files);
      await load();
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  const load = useCallback(async (resetLive = true) => {
    const d = dir.trim().replace(/^["']+|["']+$/g, "").trim();
    if (!d) return;
    setLoading(true);
    if (resetLive) onReset();
    try {
      setImages(await datasetList(d));
      setEdits({});
    } finally {
      setLoading(false);
    }
  }, [dir, onReset]);

  useEffect(() => { localStorage.setItem("soma.dir", dir); }, [dir]);
  useEffect(() => { if (cleanDir()) void load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // end of a run -> reload once from disk (the .txt captions were written)
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !caption.running) void load(false);
    wasRunning.current = caption.running;
  }, [caption.running, load]);

  const onEdit = useCallback((name: string, text: string) => {
    setEdits((s) => ({ ...s, [name]: text }));
  }, []);
  const onSave = useCallback((path: string, text: string) => {
    void saveCaption(path, text);
  }, []);

  async function start() {
    if (!images.length) await load();
    onStart({
      dataset_dir: cleanDir(),
      instance_token: token.trim(),
      model_id: "fancyfeast/llama-joycaption-beta-one-hf-llava",
      prompt: DEFAULT_PROMPT,
      max_new_tokens: 220,
      prepend_token: true,
      overwrite,
      output_dir: "",
    });
  }

  const captionedCount = useMemo(
    () => images.filter((im) => (edits[im.name] ?? caption.map[im.name] ?? im.caption).trim()).length,
    [images, edits, caption.map]
  );

  return (
    <div className="flex h-full flex-col gap-4 overflow-hidden p-4">
      <Card>
        <CardHeader title={t("ds.title")} hint={t("ds.hint")} />
        <div className="px-5 pb-2">
          {captionModel.state === "downloading" ? (
            <div className="flex items-center gap-2 text-xs text-accent">
              <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-accent" />
              <span className="shrink-0">{t("ds.modelDownloading")} {captionModel.percent}%</span>
              <div className="flex-1"><Progress value={captionModel.percent / 100} /></div>
            </div>
          ) : captionModel.state === "loading" ? (
            <div className="flex items-center gap-2 text-xs text-muted">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
              {t("ds.modelLoading")}
            </div>
          ) : modelCached ? (
            <div className="text-xs text-muted">✓ {t("ds.modelReady")}</div>
          ) : modelCached === false ? (
            <div className="text-xs text-muted">⭳ {t("ds.modelDownload")}</div>
          ) : null}
        </div>
        <div className="grid grid-cols-[1fr_auto_auto_auto] items-end gap-3 px-5 pb-3">
          <Field label={t("ds.folder")}>
            <div className="flex gap-2">
              <Input value={dir} placeholder="/root/lora01" onChange={(e) => setDir(e.target.value)} />
              {inTauri && <Button variant="ghost" onClick={() => pickFolder()}>{t("cfg.browse")}</Button>}
            </div>
          </Field>
          <Field label={t("ds.token")}>
            <Input value={token} onChange={(e) => setToken(e.target.value)} className="w-24" />
          </Field>
          <input
            ref={fileInput}
            type="file"
            multiple
            accept="image/*"
            className="hidden"
            onChange={(e) => onUpload(e.target.files)}
          />
          <Button
            variant="ghost"
            onClick={() => fileInput.current?.click()}
            disabled={uploading || !dir.trim()}
          >
            {uploading ? t("ds.uploading") : t("ds.upload")}
          </Button>
          <Button variant="ghost" onClick={() => load()} disabled={loading || !dir.trim()}>
            {loading ? t("ds.loading") : t("ds.load")}
          </Button>
        </div>
        <div className="flex items-center gap-3 px-5 pb-4">
          <label className="flex items-center gap-2 text-xs text-muted">
            <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} />
            {t("ds.overwrite")}
          </label>
          <div className="flex-1" />
          {caption.running ? (
            <Button variant="danger" onClick={onStop}>{t("ds.stop")}</Button>
          ) : (
            <Button variant="primary" onClick={start} disabled={!connected || !dir.trim()}>
              {t("ds.startCaption")}{images.length ? ` (${images.length})` : ""}
            </Button>
          )}
        </div>
      </Card>

      {/* Compact live bar: the real action happens IN the grid (the current image
          highlights + auto-scrolls + its caption is written inline). */}
      {caption.running && (
        <div className="flex items-center gap-3 px-1">
          <span className="flex shrink-0 items-center gap-1.5 text-xs text-accent">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
            {t("ds.tagging")}
          </span>
          <span className="min-w-0 flex-1 truncate text-xs text-muted">{caption.current || "…"}</span>
          <span className="shrink-0 text-xs tabular-nums text-muted">{caption.index}/{caption.total}</span>
          <div className="w-40 shrink-0">
            <Progress value={caption.total ? caption.index / caption.total : 0} />
          </div>
        </div>
      )}

      <div className="px-1 text-xs text-muted">
        {images.length} {t("ds.images")} · {captionedCount} {t("ds.captioned")}
      </div>

      <ImageGrid
        images={images}
        edits={edits}
        map={caption.map}
        current={caption.running ? caption.current : ""}
        onEdit={onEdit}
        onSave={onSave}
        t={t}
        empty={t("ds.empty")}
      />
    </div>
  );
}

const ImageGrid = memo(function ImageGrid({
  images,
  edits,
  map,
  current,
  onEdit,
  onSave,
  t,
  empty,
}: {
  images: DatasetImage[];
  edits: Record<string, string>;
  map: Record<string, string>;
  current: string;
  onEdit: (name: string, text: string) => void;
  onSave: (path: string, text: string) => void;
  t: T;
  empty: string;
}) {
  return (
    <div className="grid flex-1 grid-cols-1 gap-3 overflow-y-auto pb-2 xl:grid-cols-2">
      {images.map((img) => (
        <Row
          key={img.path}
          img={img}
          value={edits[img.name] ?? map[img.name] ?? img.caption}
          active={current === img.name}
          onEdit={onEdit}
          onSave={onSave}
          t={t}
        />
      ))}
      {!images.length && (
        <div className="col-span-full flex h-40 items-center justify-center text-sm text-muted">{empty}</div>
      )}
    </div>
  );
});

const Row = memo(function Row({
  img,
  value,
  active,
  onEdit,
  onSave,
  t,
}: {
  img: DatasetImage;
  value: string;
  active: boolean;
  onEdit: (name: string, text: string) => void;
  onSave: (path: string, text: string) => void;
  t: T;
}) {
  const ref = useRef<HTMLDivElement>(null);
  // follows the image being tagged: auto-scroll into view when the row becomes active
  useEffect(() => {
    if (active) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [active]);
  return (
    <div
      ref={ref}
      className={cn(
        "flex gap-3 rounded-xl border bg-surface p-3 transition-colors",
        active ? "border-accent ring-1 ring-accent/40" : "border-border"
      )}
    >
      <img
        src={datasetThumbUrl(img.path, 256)}
        alt={img.name}
        loading="lazy"
        className="soma-img h-32 w-24 shrink-0 rounded-lg object-cover"
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="mb-1 flex items-center justify-between gap-2">
          <span className="truncate text-xs text-muted">{img.name}</span>
          {active && <span className="shrink-0 text-xs text-accent">{t("ds.tagging")}</span>}
        </div>
        <textarea
          value={value}
          onChange={(e) => onEdit(img.name, e.target.value)}
          placeholder={t("ds.noCaption")}
          className="min-h-[90px] flex-1 resize-none rounded-lg border border-border bg-surface-2 p-2 text-xs leading-relaxed outline-none focus:border-accent"
        />
        <div className="mt-1 flex justify-end">
          <button
            onClick={() => onSave(img.path, value)}
            className="rounded-lg bg-surface-2 px-3 py-1 text-xs text-muted transition hover:text-text"
          >
            {t("ds.save")}
          </button>
        </div>
      </div>
    </div>
  );
});
