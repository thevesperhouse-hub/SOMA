import { cn } from "../lib/utils";

/** Retro XP bar: "pixel" cells + a stepped light sweep over the filled area.
 *  Everything is CSS transform/opacity -> negligible cost. */
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
  const edge = Math.floor(filled); // index of the leading (partial) cell

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
      {/* light sweep, clamped to the filled portion */}
      {p > 0.02 && (
        <div className="soma-sweep" style={{ right: `${(1 - p) * 100}%` }} />
      )}
    </div>
  );
}
