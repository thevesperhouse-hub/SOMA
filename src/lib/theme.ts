// Système de thèmes : chaque thème = un jeu complet de variables CSS (couleurs,
// polices, arrondis, densité, effets) + variantes de disposition, piloté par
// l'attribut data-theme sur <html>. Ajouter un thème = 1 entrée ici + 1 bloc CSS.
export type Theme = "forge" | "terminal" | "mono" | "editorial" | "brutal";

export const THEMES: { id: Theme; label: string; blurb: string }[] = [
  { id: "forge", label: "Forge", blurb: "Sobre · bleu · Inter" },
  { id: "terminal", label: "Terminal", blurb: "Mono · phosphore · CRT" },
  { id: "mono", label: "Mono", blurb: "Noir & blanc · clair · net" },
  { id: "editorial", label: "Éditorial", blurb: "Crème · serif · magazine" },
  { id: "brutal", label: "Brutalist", blurb: "Contraste · bordures · ombres dures" },
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
