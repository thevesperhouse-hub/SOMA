import { useEffect, useState } from "react";
import { gpuStats, type GpuStats } from "../lib/api";
import type { T } from "../lib/i18n";
import { cn } from "../lib/utils";
import { Card, CardHeader } from "./ui";

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div>
      <div className="text-[10px] font-medium uppercase tracking-wider text-muted">{label}</div>
      <div className="font-mono text-sm font-semibold tabular-nums">
        {value}
        {sub && <span className="ml-1 text-[11px] font-normal text-muted">{sub}</span>}
      </div>
    </div>
  );
}

function Bar({ pct, tone = "accent" }: { pct: number; tone?: "accent" | "warn" | "bad" }) {
  const c = tone === "bad" ? "bg-bad" : tone === "warn" ? "bg-warn" : "bg-accent";
  return (
    <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-surface-2">
      <div className={cn("h-full rounded-full transition-all duration-500", c)}
           style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
    </div>
  );
}

export function GpuMonitor({ connected, t }: { connected: boolean; t: T }) {
  const [s, setS] = useState<GpuStats | null>(null);

  useEffect(() => {
    if (!connected) return;
    let cancel = false;
    const tick = () => gpuStats().then((r) => !cancel && setS(r));
    tick();
    const id = setInterval(tick, 1500);
    return () => { cancel = true; clearInterval(id); };
  }, [connected]);

  if (!s || !s.ok) return null;

  const memPct = s.mem_total ? ((s.mem_used ?? 0) / s.mem_total) * 100 : 0;
  const memTone = memPct > 92 ? "bad" : memPct > 80 ? "warn" : "accent";
  const powPct = s.power_limit ? ((s.power ?? 0) / s.power_limit) * 100 : 0;
  const gb = (mib?: number) => ((mib ?? 0) / 1024).toFixed(1);

  return (
    <Card>
      <CardHeader
        title={s.name || "GPU"}
        hint="live"
        right={
          <span className="inline-flex items-center gap-1.5 rounded-full bg-good/15 px-2 py-0.5 text-[11px] text-good">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-good" />
            {Math.round(s.util ?? 0)}%
          </span>
        }
      />
      <div className="grid grid-cols-3 gap-x-4 gap-y-3 px-5 pb-4">
        <Metric label={t("gpu.temp")} value={`${Math.round(s.temp ?? 0)}°C`} />
        <Metric label={t("gpu.clock")} value={`${Math.round(s.clock ?? 0)}`} sub="MHz" />
        <Metric label={t("gpu.fan")} value={`${Math.round(s.fan ?? 0)}%`} />
        <div className="col-span-2">
          <Metric label={t("gpu.mem")} value={`${gb(s.mem_used)}`} sub={`/ ${gb(s.mem_total)} Go`} />
          <Bar pct={memPct} tone={memTone} />
        </div>
        <div>
          <Metric label={t("gpu.load")} value={`${Math.round(s.util ?? 0)}%`} />
          <Bar pct={s.util ?? 0} />
        </div>
        <div className="col-span-3">
          <Metric label={t("gpu.power")} value={`${Math.round(s.power ?? 0)}`} sub={`/ ${Math.round(s.power_limit ?? 0)} W`} />
          <Bar pct={powPct} tone={powPct > 92 ? "warn" : "accent"} />
        </div>
      </div>
    </Card>
  );
}
