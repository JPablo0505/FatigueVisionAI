type Status = "idle" | "ok" | "warn" | "error";

const DOT_CLASSES: Record<Status, string> = {
  idle: "h-2.5 w-2.5 rounded-full bg-slate-400",
  ok: "h-2.5 w-2.5 rounded-full bg-green-500",
  warn: "h-2.5 w-2.5 rounded-full bg-amber-500",
  error: "h-2.5 w-2.5 rounded-full bg-red-500",
};

export interface StatusIndicatorProps {
  status: Status;
  label: string;
  className?: string;
}

export function StatusIndicator({ status, label, className }: StatusIndicatorProps) {
  const classes = ["inline-flex items-center gap-2", className].filter(Boolean).join(" ");

  return (
    <div role="status" aria-live="polite" className={classes}>
      <span aria-hidden="true" className={DOT_CLASSES[status]} />
      <span className="text-sm text-slate-700">{label}</span>
    </div>
  );
}