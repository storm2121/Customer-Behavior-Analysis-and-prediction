import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";

import { keyedValue, rubberBand, Slider, snapToStep } from "./Slider";

describe("the slider's arithmetic", () => {
  it("resists more the further past the end it is pulled, and never beyond its limit", () => {
    const limit = 28;
    const near = rubberBand(10, limit);
    const far = rubberBand(100, limit);
    const farther = rubberBand(1000, limit);
    expect(near).toBeGreaterThan(0);
    expect(near).toBeLessThan(10);
    expect(far).toBeGreaterThan(near);
    expect(far - near).toBeLessThan(90);
    expect(farther).toBeLessThan(limit);
    expect(rubberBand(-10, limit)).toBeCloseTo(-near);
    expect(rubberBand(10, 0)).toBe(0);
  });

  it("snaps to the nearest step inside the range without floating-point dust", () => {
    expect(snapToStep(0.3 + 0.05, 0, 1, 0.05)).toBe(0.35);
    expect(snapToStep(0.33, 0, 1, 0.05)).toBe(0.35);
    expect(snapToStep(1.4, 0, 1, 0.05)).toBe(1);
    expect(snapToStep(-0.2, 0, 1, 0.05)).toBe(0);
    expect(snapToStep(41, 25, 65, 10)).toBe(45);
  });

  it("moves one step for an arrow, ten with Shift or a Page key, and to an end with Home or End", () => {
    expect(keyedValue("ArrowRight", false, 0.3, 0, 1, 0.05)).toBe(0.35);
    expect(keyedValue("ArrowUp", false, 0.3, 0, 1, 0.05)).toBe(0.35);
    expect(keyedValue("ArrowLeft", false, 0.3, 0, 1, 0.05)).toBe(0.25);
    expect(keyedValue("ArrowRight", true, 0.3, 0, 1, 0.05)).toBe(0.8);
    expect(keyedValue("PageDown", false, 0.3, 0, 1, 0.05)).toBe(0);
    expect(keyedValue("Home", false, 0.3, 0, 1, 0.05)).toBe(0);
    expect(keyedValue("End", false, 0.3, 0, 1, 0.05)).toBe(1);
    expect(keyedValue("a", false, 0.3, 0, 1, 0.05)).toBeNull();
  });
});

function Harness({ onChange }: { onChange: (value: number) => void }) {
  const [value, setValue] = useState(45);
  return (
    <Slider
      label="Clients younger than"
      value={value}
      min={25}
      max={65}
      step={10}
      format={(v) => `${v} years`}
      onChange={(next) => {
        setValue(next);
        onChange(next);
      }}
    />
  );
}

describe("the slider", () => {
  it("is a labelled slider that the keyboard moves by steps", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    const slider = screen.getByRole("slider", { name: "Clients younger than" });
    expect(slider).toHaveAttribute("aria-valuemin", "25");
    expect(slider).toHaveAttribute("aria-valuemax", "65");
    expect(slider).toHaveAttribute("aria-valuetext", "45 years");
    slider.focus();
    await user.keyboard("{ArrowRight}");
    expect(onChange).toHaveBeenLastCalledWith(55);
    await user.keyboard("{End}");
    expect(onChange).toHaveBeenLastCalledWith(65);
    await user.keyboard("{ArrowRight}");
    expect(onChange).toHaveBeenCalledTimes(2);
    await user.keyboard("{Shift>}{ArrowLeft}{/Shift}");
    expect(onChange).toHaveBeenLastCalledWith(25);
    expect(slider).toHaveAttribute("aria-valuenow", "25");
  });
});
