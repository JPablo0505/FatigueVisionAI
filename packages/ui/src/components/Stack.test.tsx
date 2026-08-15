import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Stack } from "../components/Stack";

describe("Stack", () => {
  it("renderiza los children", () => {
    render(<Stack>Uno</Stack>);
    expect(screen.getByText("Uno")).toBeInTheDocument();
  });

  it("aplica gap-0 cuando gap es 0", () => {
    const { container } = render(
      <Stack gap={0}>
        <span>a</span>
        <span>b</span>
      </Stack>
    );
    expect(container.querySelector<HTMLElement>(".gap-0")).not.toBeNull();
  });
});