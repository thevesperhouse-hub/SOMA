import { useEffect, useRef } from "react";
import type { Sample } from "../types";
import { cn } from "../lib/utils";

// Dessine un aperçu "personnage" procédural qui se précise avec `sharpness`
// (0 = bruit/flou, 1 = net). Sert au mode démo pour matérialiser le
// "watch your LoRA learn" sans GPU ni modèle.
function drawProcedural(canvas: HTMLCanvasElement, seed: number, sharpness: number) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = (canvas.width = 320);
  const h = (canvas.height = 320);

  let s = seed * 9301 + 49297;
  const rnd = () => ((s = (s * 9301 + 49297) % 233280) / 233280);

  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, "#23262e");
  bg.addColorStop(1, "#15171c");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  const cx = w / 2;
  const skin = `rgba(${(180 + rnd() * 30) | 0}, ${(150 + rnd() * 30) | 0}, ${(140 + rnd() * 20) | 0}, ${0.25 + 0.7 * sharpness})`;
  ctx.fillStyle = skin;
  ctx.beginPath();
  ctx.ellipse(cx, h * 0.92, w * 0.34, h * 0.22, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(cx, h * 0.42, w * 0.17, h * 0.21, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = `rgba(30,30,35,${0.15 + 0.65 * sharpness})`;
  ctx.beginPath();
  ctx.ellipse(cx - 22, h * 0.4, 6, 8 * (0.4 + sharpness), 0, 0, Math.PI * 2);
  ctx.ellipse(cx + 22, h * 0.4, 6, 8 * (0.4 + sharpness), 0, 0, Math.PI * 2);
  ctx.fill();

  const noise = Math.max(0, 1 - sharpness);
  const count = Math.floor(noise * 9000);
  for (let i = 0; i < count; i++) {
    const x = rnd() * w;
    const y = rnd() * h;
    const a = rnd() * noise * 0.5;
    ctx.fillStyle = `rgba(${(rnd() * 255) | 0},${(rnd() * 255) | 0},${(rnd() * 255) | 0},${a})`;
    ctx.fillRect(x, y, 2, 2);
  }
}

function Canvas({ sample, className }: { sample: Sample; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!sample.placeholder && sample.image) return;
    if (ref.current) drawProcedural(ref.current, sample.seed, sample.sharpness);
  }, [sample]);
  if (!sample.placeholder && sample.image) {
    return <img src={sample.image} alt="" className={cn("object-cover", className)} />;
  }
  return <canvas ref={ref} className={className} />;
}

export function SampleViewer({ samples }: { samples: Sample[] }) {
  const latest = samples[samples.length - 1];
  return (
    <div className="px-5 pb-5">
      <div className="aspect-square w-full overflow-hidden rounded-xl border border-border bg-surface-2">
        {latest ? (
          <Canvas sample={latest} className="h-full w-full rounded-xl animate-pop" />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-muted">
            Les aperçus apparaîtront ici
          </div>
        )}
      </div>
      {latest && (
        <div className="mt-2 flex items-center justify-between text-xs text-muted">
          <span>step {latest.step}</span>
          <span>netteté {Math.round(latest.sharpness * 100)}%</span>
        </div>
      )}
      {samples.length > 1 && (
        <div className="mt-3 flex gap-2 overflow-x-auto pb-1">
          {samples.slice(-8).map((s, i) => (
            <div
              key={i}
              className={cn(
                "h-12 w-12 shrink-0 overflow-hidden rounded-lg border",
                s === latest ? "border-accent" : "border-border"
              )}
            >
              <Canvas sample={s} className="h-full w-full" />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
