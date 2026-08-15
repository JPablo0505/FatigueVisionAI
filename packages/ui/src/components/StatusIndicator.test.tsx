import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { StatusIndicator } from "../components/StatusIndicator";

describe("StatusIndicator", () => {
  it("se renderiza con role status y el label visible", () => {
    render(<StatusIndicator status="ok" label="Conectado" />);
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent("Conectado");
  });

  it("aplica aria-live polite", () => {
    render(<StatusIndicator status="error" label="Sin conexión" />);
    expect(screen.getByRole("status")).toHaveAttribute("aria-live", "polite");
  });
});