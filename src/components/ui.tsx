import { cn } from "../lib/utils";
import type { ReactNode } from "react";

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div className={cn("soma-card rounded-2xl border border-border bg-surface", className)}>
      {children}
    </div>
  );
}

export function CardHeader({ title, hint, right }: { title: string; hint?: string; right?: ReactNode }) {
  return (
    <div className="flex items-center justify-between px-5 pt-4 pb-3">
      <div className="flex items-stretch gap-2.5">
        <span className="soma-tick my-0.5" />
        <div>
          <h3 className="soma-cardtitle text-sm font-semibold tracking-tight">{title}</h3>
          {hint && <p className="text-xs text-muted mt-0.5">{hint}</p>}
        </div>
      </div>
      {right}
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "primary",
  disabled,
  className,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost" | "danger";
  disabled?: boolean;
  className?: string;
}) {
  const styles = {
    primary: "bg-accent text-white hover:brightness-110 soma-sheen soma-accent-glow soma-btn-primary",
    ghost: "bg-surface-2 text-text hover:bg-border",
    danger: "bg-bad/15 text-bad hover:bg-bad/25",
  }[variant];
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "soma-btn inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-medium transition",
        "active:translate-y-px disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none",
        styles,
        className
      )}
    >
      {children}
    </button>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: string }) {
  return (
    <div className="soma-card rounded-xl border border-border bg-surface-2 px-4 py-3">
      <div className="text-[11px] font-medium uppercase tracking-wider text-muted">{label}</div>
      <div className="mt-1 font-mono text-xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="mt-0.5 font-mono text-[11px] text-muted">{sub}</div>}
    </div>
  );
}

export function Progress({ value }: { value: number }) {
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2">
      <div
        className="h-full rounded-full bg-accent transition-all duration-300"
        style={{ width: `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%` }}
      />
    </div>
  );
}

export function Badge({ children, tone = "muted" }: { children: ReactNode; tone?: "muted" | "good" | "warn" | "bad" | "accent" }) {
  const tones = {
    muted: "bg-surface-2 text-muted",
    good: "bg-good/15 text-good",
    warn: "bg-warn/15 text-warn",
    bad: "bg-bad/15 text-bad",
    accent: "bg-accent-soft text-accent",
  }[tone];
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium", tones)}>
      {children}
    </span>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-muted">{label}</span>
      <div className="mt-1">{children}</div>
      {hint && <span className="text-[11px] text-muted">{hint}</span>}
    </label>
  );
}

export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        "w-full rounded-xl border border-border bg-surface-2 px-3 py-2 text-sm outline-none",
        "focus:border-accent focus:ring-2 focus:ring-accent-soft transition",
        props.className
      )}
    />
  );
}
