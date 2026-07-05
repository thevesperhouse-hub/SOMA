// Couche gamification : niveaux + succès calculés à partir des stats du run.
// Volontairement côté client (le moteur reste pur) et désactivable.

export interface GameStats {
  steps: number;
  totalSteps: number;
  bestLoss: number;
  lastLoss: number;
  firstLoss: number;
  samples: number;
  finished: boolean;
}

export interface Achievement {
  id: string;
  title: string;
  desc: string;
  icon: string;
  test: (s: GameStats) => boolean;
}

export const ACHIEVEMENTS: Achievement[] = [
  { id: "first_step", title: "Premier pas", desc: "Lancer un entraînement", icon: "✦",
    test: (s) => s.steps >= 1 },
  { id: "warmup", title: "Échauffement", desc: "Atteindre 100 steps", icon: "▲",
    test: (s) => s.steps >= 100 },
  { id: "first_sample", title: "Première vision", desc: "Générer un aperçu live", icon: "◎",
    test: (s) => s.samples >= 1 },
  { id: "convergence", title: "Ça converge", desc: "Loss divisée par 2", icon: "↘",
    test: (s) => s.firstLoss > 0 && s.bestLoss <= s.firstLoss / 2 },
  { id: "sharp", title: "Identity Lock", desc: "Loss sous 0.06", icon: "◆",
    test: (s) => s.bestLoss > 0 && s.bestLoss < 0.06 },
  { id: "marathon", title: "Marathon", desc: "Atteindre 1000 steps", icon: "⬢",
    test: (s) => s.steps >= 1000 },
  { id: "finisher", title: "Forgeron", desc: "Terminer un entraînement", icon: "★",
    test: (s) => s.finished },
];

export function unlockedAchievements(s: GameStats): string[] {
  return ACHIEVEMENTS.filter((a) => a.test(s)).map((a) => a.id);
}

export function computeXp(s: GameStats, unlockedCount: number): number {
  return Math.round(s.steps + s.samples * 25 + unlockedCount * 60 + (s.finished ? 200 : 0));
}

function xpForLevel(level: number): number {
  // cumul croissant, doux au début puis plus exigeant
  return Math.round(120 * Math.pow(level, 1.5));
}

export interface LevelInfo {
  level: number;
  intoLevel: number;
  span: number;
  pct: number;
}

export function levelFromXp(xp: number): LevelInfo {
  let level = 1;
  while (xp >= xpForLevel(level + 1)) level++;
  const base = xpForLevel(level);
  const next = xpForLevel(level + 1);
  const span = Math.max(1, next - base);
  const intoLevel = Math.max(0, xp - base);
  return { level, intoLevel, span, pct: Math.min(1, intoLevel / span) };
}
