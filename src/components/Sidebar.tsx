import type { ReactNode } from "react";
import type { T } from "../lib/i18n";
import { cn } from "../lib/utils";
import { Logo } from "./Logo";

export type View = "dashboard" | "datasets" | "models" | "settings";

const ICONS: Record<View, ReactNode> = {
  dashboard: (
    <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="7" height="9" rx="1.5" /><rect x="14" y="3" width="7" height="5" rx="1.5" />
      <rect x="14" y="12" width="7" height="9" rx="1.5" /><rect x="3" y="16" width="7" height="5" rx="1.5" />
    </svg>
  ),
  datasets: (
    <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="m21 15-5-5L5 21" />
    </svg>
  ),
  models: (
    <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2 3 7v10l9 5 9-5V7l-9-5Z" /><path d="M3 7l9 5 9-5M12 22V12" />
    </svg>
  ),
  settings: (
    <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M19.1 4.9l-1.8 1.8M6.7 17.3l-1.8 1.8" />
    </svg>
  ),
};

export function Sidebar({
  view,
  setView,
  connected,
  t,
}: {
  view: View;
  setView: (v: View) => void;
  connected: boolean;
  t: T;
}) {
  const items: { id: View; label: string }[] = [
    { id: "dashboard", label: t("nav.dashboard") },
    { id: "datasets", label: t("nav.datasets") },
    { id: "models", label: t("nav.models") },
    { id: "settings", label: t("nav.settings") },
  ];
  return (
    <aside className="soma-sidebar flex w-56 shrink-0 flex-col border-r border-border bg-surface/60">
      <div className="flex items-center gap-3 px-5 py-5">
        <div className="soma-logo-tile relative grid h-9 w-9 place-items-center rounded-xl border border-accent/30 bg-accent/10">
          <div className="absolute inset-0 rounded-xl bg-accent/20 blur-md" />
          <Logo className="relative h-5 w-5" />
        </div>
        <div>
          <div className="soma-wordmark text-sm font-bold leading-none">SOMA</div>
          <div className="mt-1 text-[10px] uppercase tracking-wider text-muted">{t("tagline")}</div>
        </div>
      </div>

      <nav className="flex flex-1 flex-col gap-1 px-3">
        {items.map((it) => (
          <button
            key={it.id}
            onClick={() => setView(it.id)}
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition",
              view === it.id
                ? "bg-accent-soft text-accent"
                : "text-muted hover:bg-surface-2 hover:text-text"
            )}
          >
            {ICONS[it.id]}
            {it.label}
          </button>
        ))}
      </nav>

      <div className="flex items-center gap-2 px-5 py-4 text-[11px] text-muted">
        <span className={cn("h-1.5 w-1.5 rounded-full", connected ? "bg-good" : "bg-bad")} />
        {connected ? t("cfg.engineOk") : t("cfg.offline")}
      </div>
    </aside>
  );
}
