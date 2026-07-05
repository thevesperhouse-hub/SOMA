import { useEffect, useState } from "react";
import { downloadUrl, listCheckpoints, type Checkpoint } from "../lib/api";
import type { T } from "../lib/i18n";
import { cn } from "../lib/utils";

function fmtDate(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  return isNaN(+d) ? "" : d.toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "2-digit" });
}

const ModelIcon = () => (
  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 2 3 7v10l9 5 9-5V7l-9-5Z" /><path d="M3 7l9 5 9-5" />
  </svg>
);
const DownloadIcon = () => (
  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3v12m0 0 4-4m-4 4-4-4M4 21h16" />
  </svg>
);

export function ModelsView({
  connected,
  outputDir,
  refreshKey,
  t,
}: {
  connected: boolean;
  outputDir: string;
  refreshKey: number;
  t: T;
}) {
  const [items, setItems] = useState<Checkpoint[]>([]);

  useEffect(() => {
    if (!connected) return;
    let cancel = false;
    listCheckpoints(outputDir || "output").then((c) => !cancel && setItems(c));
    return () => { cancel = true; };
  }, [connected, outputDir, refreshKey]);

  // groupe par archi (label), "Archi inconnue" en dernier
  const groups = new Map<string, Checkpoint[]>();
  for (const c of items) {
    const key = c.label || c.arch || t("models.unknown");
    (groups.get(key) ?? groups.set(key, []).get(key)!).push(c);
  }

  return (
    <div className="mx-auto flex h-full w-full max-w-4xl flex-col gap-6 overflow-y-auto p-6">
      <div>
        <h2 className="soma-wordmark text-lg font-bold">{t("models.title")}</h2>
        <p className="mt-1 text-xs text-muted">{items.length} LoRA</p>
      </div>

      {items.length === 0 && (
        <div className="flex h-40 items-center justify-center rounded-xl border border-dashed border-border text-sm text-muted">
          {t("models.empty")}
        </div>
      )}

      {[...groups.entries()].map(([label, list]) => (
        <section key={label} className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <span className="soma-tick h-3.5 w-1" />
            <h3 className="text-sm font-semibold">{label}</h3>
            <span className="rounded-full bg-surface-2 px-2 py-0.5 text-[11px] text-muted">{list.length}</span>
          </div>
          {list.map((c) => (
            <div
              key={c.path}
              className="soma-card flex items-center gap-3 rounded-xl border border-border bg-surface px-4 py-3"
            >
              <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-accent-soft text-accent">
                <ModelIcon />
              </span>
              <div className="min-w-0 flex-1">
                <div className="truncate font-mono text-sm">{c.name}</div>
                <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-muted">
                  {c.base && <span className="truncate">{c.base}</span>}
                  {c.steps && <span>· {c.steps} {t("models.steps")}</span>}
                  {c.date && <span>· {fmtDate(c.date)}</span>}
                  <span className="tabular-nums">· {c.size_mb} Mo</span>
                </div>
              </div>
              <a
                href={downloadUrl(c.path)}
                download={c.name}
                className={cn(
                  "grid h-8 w-8 shrink-0 place-items-center rounded-lg border border-border text-muted transition",
                  "hover:border-accent/50 hover:text-accent"
                )}
                title="Download"
              >
                <DownloadIcon />
              </a>
            </div>
          ))}
        </section>
      ))}
    </div>
  );
}
