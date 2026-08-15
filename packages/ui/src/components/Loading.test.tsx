import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Loading } from "../components/Loading";

describe("Loading", () => {
  it("se renderiza con role status y aria-busy true", () => {
    render(<Loading />);
    const status = screen.getByRole("status");
    expect(status).toHaveAttribute("aria-busy", "true");
  });

  it("muestra el label por defecto", () => {
    render(<Loading />);
    expect(screen.getByText("Cargando...")).toBeInTheDocument();
  });
});