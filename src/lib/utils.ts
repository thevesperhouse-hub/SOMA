export function cn(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

export function fmtSecs(s: number): string {
  if (!isFinite(s)) return "—";
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return m > 0 ? `${m}m ${r}s` : `${r}s`;
}

export function fmtLr(lr: number): string {
  return lr ? lr.toExponential(1) : "—";
}
