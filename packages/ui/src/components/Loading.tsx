export interface LoadingProps {
  label?: string;
  className?: string;
}

export function Loading({ label = "Cargando...", className }: LoadingProps) {
  const classes = ["inline-flex items-center gap-2", className].filter(Boolean).join(" ");

  return (
    <div role="status" aria-busy="true" className={classes}>
      <span
        aria-hidden="true"
        className="h-5 w-5 animate-spin rounded-full border-2 border-slate-300 border-t-slate-700"
      />
      <span className="text-sm text-slate-700">{label}</span>
    </div>
  );
}