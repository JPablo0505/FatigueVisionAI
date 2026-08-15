import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Alert } from "../components/Alert";

describe("Alert", () => {
  it("usa role alert cuando tone es error", () => {
    render(<Alert tone="error">Algo salió mal</Alert>);
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("usa role status con aria-live polite en tone info", () => {
    render(<Alert tone="info">Proceso completado</Alert>);
    const status = screen.getByRole("status");
    expect(status).toHaveAttribute("aria-live", "polite");
  });

  it("renderiza el title en strong", () => {
    render(<Alert title="Aviso">Texto del aviso</Alert>);
    expect(screen.getByRole("status")).toContainElement(screen.getByText("Aviso"));
  });
});