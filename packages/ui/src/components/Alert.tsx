import type { ReactNode } from "react";

type AlertTone = "info" | "success" | "warning" | "error";

const TONES: Record<AlertTone, string> = {
  info: "border-blue-200 bg-blue-50 text-blue-900",
  success: "border-green-200 bg-green-50 text-green-900",
  warning: "border-amber-200 bg-amber-50 text-amber-900",
  error: "border-red-200 bg-red-50 text-red-900",
};

export interface AlertProps {
  tone?: AlertTone;
  title?: string;
  children: ReactNode;
  className?: string;
}

export function Alert({ tone = "info", title, children, className }: AlertProps) {
  const isError = tone === "error";
  const role = isError ? "alert" : "status";
  const classes = ["rounded-lg border p-4 text-sm", TONES[tone], className].filter(Boolean).join(" ");

  return (
    <div role={role} aria-live={isError ? undefined : "polite"} className={classes}>
      {title ? <strong className="mb-1 block font-semibold">{title}</strong> : null}
      {children}
    </div>
  );
}