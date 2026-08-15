import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Badge } from "../components/Badge";

describe("Badge", () => {
  it("existe con children", () => {
    render(<Badge>Beta</Badge>);
    expect(screen.getByText("Beta")).toBeInTheDocument();
  });

  it("aplica ariaLabel cuando se provee", () => {
    render(<Badge ariaLabel="Conexión estable">OK</Badge>);
    expect(screen.getByText("OK")).toHaveAttribute("aria-label", "Conexión estable");
  });
});