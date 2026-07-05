import { cn } from "../lib/utils";

/** Barre d'XP rétro : cellules "pixel" + balayage lumineux cranté sur la zone
 *  remplie. Tout est en CSS transform/opacity -> coût négligeable. */
export function PixelBar({
  pct,
  cells = 36,
  className,
  height = "h-3.5",
}: {
  pct: number;
  cells?: number;
  className?: string;
  height?: string;
}) {
  const p = Math.max(0, Math.min(1, pct));
  const filled = p * cells;
  const edge = Math.floor(filled); // index de la cellule de tête (partielle)

  return (
    <div className={cn("relative", className)}>
      <div className={cn("flex gap-[3px]", height)}>
        {Array.from({ length: cells }).map((_, i) => {
          const isFilled = i < edge;
          const isEdge = i === edge && p < 1 && p > 0;
          return (
            <div
              key={i}
              className={cn(
                "flex-1 rounded-[2px] transition-colors duration-300",
                isFilled
                  ? "bg-accent"
                  : isEdge
                  ? "bg-accent/60 soma-blink"
                  : "bg-surface-2"
              )}
            />
          );
        })}
      </div>
      {/* balayage lumineux, borné à la portion remplie */}
      {p > 0.02 && (
        <div className="soma-sweep" style={{ right: `${(1 - p) * 100}%` }} />
      )}
    </div>
  );
}
