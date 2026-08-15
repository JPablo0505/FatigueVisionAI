import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Button } from "../index";

describe("Button", () => {
  it("se renderiza con rol button", () => {
    render(<Button>Guardar</Button>);
    expect(screen.getByRole("button", { name: "Guardar" })).toBeInTheDocument();
  });

  it("dispara onClick al hacer click", () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Aceptar</Button>);
    fireEvent.click(screen.getByRole("button"));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("deshabilita el botón cuando disabled", () => {
    render(<Button disabled>Deshabilitado</Button>);
    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("aplica el type prop", () => {
    render(<Button type="submit">Enviar</Button>);
    expect(screen.getByRole("button")).toHaveAttribute("type", "submit");
  });
});