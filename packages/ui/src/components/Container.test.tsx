import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Container } from "../components/Container";

describe("Container", () => {
  it("renderiza los children", () => {
    render(<Container>Contenido del contenedor</Container>);
    expect(screen.getByText("Contenido del contenedor")).toBeInTheDocument();
  });
});