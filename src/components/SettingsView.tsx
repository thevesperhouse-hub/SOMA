import type { Lang, T } from "../lib/i18n";
import { THEMES, type Theme } from "../lib/theme";
import { cn } from "../lib/utils";

export function SettingsView({
  theme,
  setTheme,
  lang,
  setLang,
  gameOn,
  setGameOn,
  t,
}: {
  theme: Theme;
  setTheme: (t: Theme) => void;
  lang: Lang;
  setLang: (l: Lang) => void;
  gameOn: boolean;
  setGameOn: (b: boolean) => void;
  t: T;
}) {
  return (
    <div className="mx-auto flex h-full w-full max-w-2xl flex-col gap-8 overflow-y-auto p-6">
      <h2 className="soma-wordmark text-lg font-bold">{t("nav.settings")}</h2>

      <section>
        <div className="mb-2 text-xs font-medium uppercase tracking-wider text-muted">{t("settings.theme")}</div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {THEMES.map((th) => (
            <button
              key={th.id}
              onClick={() => setTheme(th.id)}
              className={cn(
                "soma-card flex flex-col rounded-xl border px-4 py-3 text-left transition",
                theme === th.id ? "border-accent bg-accent-soft" : "border-border bg-surface hover:border-accent/40"
              )}
            >
              <span className="text-sm font-medium">{th.label}</span>
              <span className="mt-0.5 text-[11px] text-muted">{th.blurb}</span>
            </button>
          ))}
        </div>
      </section>

      <section>
        <div className="mb-2 text-xs font-medium uppercase tracking-wider text-muted">{t("settings.language")}</div>
        <div className="flex gap-2">
          {(["fr", "en"] as Lang[]).map((l) => (
            <button
              key={l}
              onClick={() => setLang(l)}
              className={cn(
                "rounded-lg border px-5 py-2 text-sm font-medium transition",
                lang === l ? "border-accent bg-accent-soft text-accent" : "border-border bg-surface-2 text-muted hover:text-text"
              )}
            >
              {l.toUpperCase()}
            </button>
          ))}
        </div>
      </section>

      <section>
        <div className="mb-2 text-xs font-medium uppercase tracking-wider text-muted">{t("game.title")}</div>
        <label className="soma-card flex max-w-sm items-center justify-between rounded-xl border border-border bg-surface px-4 py-3">
          <div>
            <div className="text-sm font-medium">{t("game.hint")}</div>
            <div className="text-xs text-muted">{gameOn ? "ON" : "OFF"}</div>
          </div>
          <button
            onClick={() => setGameOn(!gameOn)}
            className={cn("relative h-6 w-11 rounded-full transition", gameOn ? "bg-accent" : "bg-border")}
          >
            <span className={cn("absolute top-0.5 h-5 w-5 rounded-full bg-white transition-all", gameOn ? "left-[22px]" : "left-0.5")} />
          </button>
        </label>
      </section>
    </div>
  );
}
