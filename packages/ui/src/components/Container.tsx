import type { ReactNode } from "react";

type MaxWidth = "sm" | "md" | "lg" | "xl";

const MAX_WIDTH: Record<MaxWidth, string> = {
  sm: "max-w-sm",
  md: "max-w-md",
  lg: "max-w-lg",
  xl: "max-w-xl",
};

export interface ContainerProps {
  maxWidth?: MaxWidth;
  children: ReactNode;
  className?: string;
}

export function Container({ maxWidth = "lg", children, className }: ContainerProps) {
  const classes = ["mx-auto w-full px-4", MAX_WIDTH[maxWidth], className].filter(Boolean).join(" ");

  return <div className={classes}>{children}</div>;
}