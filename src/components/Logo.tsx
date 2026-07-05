// Marque SOMA — monogramme "S" organique sur tuile, dans la couleur d'accent.
export function Logo({ className = "h-8 w-8" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} role="img" aria-label="SOMA">
      <rect width="32" height="32" rx="8" fill="var(--accent)" />
      <path
        d="M21.2 10.8c-1.7-1.5-4.2-2-6.2-1.3-2.3.8-3 3.1-1.4 4.6 1.4 1.3 4.1 1.2 5.7 2.6 1.7 1.5.8 3.9-1.5 4.6-2.1.7-4.6.1-6.2-1.4"
        fill="none"
        stroke="#fff"
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
