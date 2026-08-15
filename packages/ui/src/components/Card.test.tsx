import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Card } from "../components/Card";

describe("Card", () => {
  it("muestra el title como heading", () => {
    render(<Card title="Estado del sistema">Contenido</Card>);
    expect(screen.getByRole("heading", { name: "Estado del sistema" })).toBeInTheDocument();
  });

  it("renderiza los children", () => {
    render(<Card>Contenido sin título</Card>);
    expect(screen.getByText("Contenido sin título")).toBeInTheDocument();
  });
});