import type { ReactNode } from "react";

type Tone = "neutral" | "info" | "success" | "warning" | "error";

const TONES: Record<Tone, string> = {
  neutral: "bg-slate-100 text-slate-700",
  info: "bg-blue-100 text-blue-800",
  success: "bg-green-100 text-green-800",
  warning: "bg-amber-100 text-amber-800",
  error: "bg-red-100 text-red-800",
};

export interface BadgeProps {
  tone?: Tone;
  children: ReactNode;
  className?: string;
  ariaLabel?: string;
}

export function Badge({ tone = "neutral", children, className, ariaLabel }: BadgeProps) {
  const classes = [
    "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
    TONES[tone],
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <span className={classes} aria-label={ariaLabel}>
      {children}
    </span>
  );
}