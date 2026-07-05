import { ACHIEVEMENTS, levelFromXp } from "../lib/game";
import type { T } from "../lib/i18n";
import { cn } from "../lib/utils";
import { Card, CardHeader } from "./ui";

/** Roue d'XP : jauge conique segmentée (encoches) + halo accent, anneau interne
 *  fin pour le run en cours. Progression animée en douceur (SVG dashoffset). */
function XpWheel({
  globalPct,
  runPct,
  level,
  label,
}: {
  globalPct: number;
  runPct: number;
  level: number;
  label: string;
}) {
  const R = 33;
  const C = 2 * Math.PI * R;
  const Ri = R - 8.5;
  const Ci = 2 * Math.PI * Ri;
  const segments = 44;
  const notch = 2.1; // largeur d'encoche (unités de path)
  return (
    <div className="relative h-28 w-28 shrink-0">
      <svg viewBox="0 0 80 80" className="h-full w-full -rotate-90">
        <defs>
          <linearGradient id="xpgrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="var(--accent)" />
            <stop offset="1" stopColor="var(--accent)" stopOpacity="0.55" />
          </linearGradient>
        </defs>
        {/* piste externe */}
        <circle cx="40" cy="40" r={R} fill="none" stroke="var(--surface-2)" strokeWidth="8" />
        {/* anneau interne : run en cours */}
        <circle cx="40" cy="40" r={Ri} fill="none" stroke="var(--surface-2)" strokeWidth="3" />
        <circle
          cx="40" cy="40" r={Ri} fill="none"
          stroke="var(--accent)" strokeOpacity="0.5" strokeWidth="3" strokeLinecap="round"
          strokeDasharray={Ci} strokeDashoffset={Ci * (1 - runPct)}
          style={{ transition: "stroke-dashoffset .5s ease" }}
        />
        {/* progression globale + halo */}
        <circle
          cx="40" cy="40" r={R} fill="none"
          stroke="url(#xpgrad)" strokeWidth="8" strokeLinecap="round"
          strokeDasharray={C} strokeDashoffset={C * (1 - globalPct)}
          style={{
            transition: "stroke-dashoffset .6s cubic-bezier(.2,.8,.2,1)",
            filter: "drop-shadow(0 0 2.5px var(--accent))",
          }}
        />
        {/* encoches : dashes couleur carte par-dessus -> jauge segmentée */}
        <circle
          cx="40" cy="40" r={R} fill="none" stroke="var(--surface)" strokeWidth="9"
          strokeDasharray={`${notch} ${C / segments - notch}`}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-[9px] font-medium uppercase tracking-[0.18em] text-muted">{label}</span>
        <span className="text-[26px] font-bold leading-none tabular-nums">{level}</span>
      </div>
    </div>
  );
}

export function GamePanel({
  runXp,
  globalXp,
  unlocked,
  enabled,
  onToggle,
  t,
}: {
  runXp: number;
  globalXp: number;
  unlocked: Set<string>;
  enabled: boolean;
  onToggle: () => void;
  t: T;
}) {
  const g = levelFromXp(globalXp);
  const r = levelFromXp(runXp);
  return (
    <Card className="flex h-full flex-col">
      <CardHeader
        title={t("game.title")}
        hint={t("game.hint")}
        right={
          <button
            onClick={onToggle}
            className={cn(
              "rounded-full px-2.5 py-1 text-xs font-medium transition",
              enabled ? "bg-accent-soft text-accent" : "bg-surface-2 text-muted"
            )}
          >
            {enabled ? "ON" : "OFF"}
          </button>
        }
      />
      {enabled ? (
        <div className="flex flex-1 flex-col gap-4 px-5 pb-5">
          {/* XP globale (persistante) */}
          <div className="flex items-center gap-4">
            <XpWheel globalPct={g.pct} runPct={r.pct} level={g.level} label={t("game.level")} />
            <div className="min-w-0 flex-1">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted">
                {t("game.global")}
              </div>
              <div className="mt-0.5 text-2xl font-bold leading-none tabular-nums">
                {globalXp.toLocaleString()}
                <span className="ml-1 text-sm font-medium text-muted">XP</span>
              </div>
              <div className="mt-2 flex items-center justify-between text-xs text-muted">
                <span>
                  {t("game.thisLora")} <span className="font-semibold text-accent">+{runXp.toLocaleString()}</span>
                </span>
                <span className="tabular-nums">
                  {unlocked.size}/{ACHIEVEMENTS.length}
                </span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-2 overflow-y-auto">
            {ACHIEVEMENTS.map((a) => {
              const on = unlocked.has(a.id);
              return (
                <div
                  key={a.id}
                  className={cn(
                    "flex items-center gap-3 rounded-xl border px-3 py-2.5 transition",
                    on
                      ? "border-accent/40 bg-accent-soft animate-slidein"
                      : "border-border bg-surface-2 opacity-60"
                  )}
                >
                  <div
                    className={cn(
                      "flex h-8 w-8 items-center justify-center rounded-lg text-base",
                      on ? "bg-accent text-white" : "bg-border text-muted"
                    )}
                  >
                    {a.icon}
                  </div>
                  <div className="min-w-0">
                    <div className="text-sm font-medium">{t(`ach.${a.id}.title`)}</div>
                    <div className="truncate text-xs text-muted">{t(`ach.${a.id}.desc`)}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <div className="flex flex-1 items-center justify-center px-5 pb-5 text-center text-sm text-muted">
          {t("game.offTitle")}
          <br />
          {t("game.offSub")}
        </div>
      )}
    </Card>
  );
}
