import { useMemo } from "react";

// Loss curve in pure SVG (no external dep), crisp and plain.
export function LossChart({ losses }: { losses: number[] }) {
  const W = 640;
  const H = 200;
  const pad = 8;

  const { path, area, min, max, last } = useMemo(() => {
    if (losses.length < 2) {
      return { path: "", area: "", min: 0, max: 0, last: losses[0] ?? 0 };
    }
    const lo = Math.min(...losses);
    const hi = Math.max(...losses);
    const span = hi - lo || 1;
    const n = losses.length;
    const x = (i: number) => pad + (i / (n - 1)) * (W - pad * 2);
    const y = (v: number) => pad + (1 - (v - lo) / span) * (H - pad * 2);
    const pts = losses.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
    return {
      path: "M" + pts.join(" L"),
      area: `M${x(0)},${H - pad} L` + pts.join(" L") + ` L${x(n - 1)},${H - pad} Z`,
      min: lo,
      max: hi,
      last: losses[n - 1],
    };
  }, [losses]);

  return (
    <div className="px-5 pb-5">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none" style={{ height: 200 }}>
        <defs>
          <linearGradient id="lossfill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.22" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
          </linearGradient>
        </defs>
        {area && <path d={area} fill="url(#lossfill)" />}
        {path && <path d={path} fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinejoin="round" />}
        {losses.length < 2 && (
          <text x={W / 2} y={H / 2} textAnchor="middle" fill="var(--muted)" fontSize="13">
            Waiting for data…
          </text>
        )}
      </svg>
      <div className="mt-2 flex items-center justify-between text-xs text-muted tabular-nums">
        <span>min {min ? min.toFixed(4) : "—"}</span>
        <span>current <span className="text-text font-medium">{last ? last.toFixed(4) : "—"}</span></span>
        <span>max {max ? max.toFixed(4) : "—"}</span>
      </div>
    </div>
  );
}
