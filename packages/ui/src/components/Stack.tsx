import type { ReactNode } from "react";

type Direction = "column" | "row";
type Gap = 0 | 1 | 2 | 3 | 4 | 6 | 8;
type Align = "start" | "center" | "end" | "stretch";

const GAPS: Record<Gap, string> = {
  0: "gap-0",
  1: "gap-1",
  2: "gap-2",
  3: "gap-3",
  4: "gap-4",
  6: "gap-6",
  8: "gap-8",
};

const ALIGNS: Record<Align, string> = {
  start: "items-start",
  center: "items-center",
  end: "items-end",
  stretch: "items-stretch",
};

export interface StackProps {
  direction?: Direction;
  gap?: Gap;
  align?: Align;
  children: ReactNode;
  className?: string;
}

export function Stack({
  direction = "column",
  gap = 4,
  align = "stretch",
  children,
  className,
}: StackProps) {
  const classes = [
    direction === "column" ? "flex flex-col" : "flex flex-row",
    GAPS[gap],
    ALIGNS[align],
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return <div className={classes}>{children}</div>;
}