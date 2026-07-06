// Theme system: each theme = a full set of CSS variables (colors, fonts, radii,
// density, effects) + layout variants, driven by the data-theme attribute on <html>.
// Adding a theme = one entry here + one CSS block.
export type Theme = "forge" | "terminal" | "mono" | "editorial" | "brutal";

export const THEMES: { id: Theme; label: string; blurb: string }[] = [
  { id: "forge", label: "Forge", blurb: "Sober · blue · Inter" },
  { id: "terminal", label: "Terminal", blurb: "Mono · phosphor · CRT" },
  { id: "mono", label: "Mono", blurb: "Black & white · light · crisp" },
  { id: "editorial", label: "Editorial", blurb: "Cream · serif · magazine" },
  { id: "brutal", label: "Brutalist", blurb: "Contrast · borders · hard shadows" },
];

const IDS = new Set(THEMES.map((t) => t.id));

export function getInitialTheme(): Theme {
  const s = localStorage.getItem("soma.theme") as Theme | null;
  return s && IDS.has(s) ? s : "forge";
}

export function applyTheme(t: Theme) {
  document.documentElement.dataset.theme = t;
  localStorage.setItem("soma.theme", t);
}
