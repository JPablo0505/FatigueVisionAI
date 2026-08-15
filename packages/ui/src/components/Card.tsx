import type { ReactNode } from "react";

export interface CardProps {
  title?: string;
  children: ReactNode;
  className?: string;
}

export function Card({ title, children, className }: CardProps) {
  const classes = ["rounded-2xl border border-slate-200 bg-white p-6 shadow-sm", className]
    .filter(Boolean)
    .join(" ");

  return (
    <section className={classes}>
      {title ? <h2 className="mb-4 text-lg font-semibold text-slate-900">{title}</h2> : null}
      {children}
    </section>
  );
}