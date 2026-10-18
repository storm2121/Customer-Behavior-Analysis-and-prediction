import type { Meta, StoryObj } from "@storybook/react";
import { useState } from "react";

import { Slider } from "../components/Controls";
import { Equation } from "../components/Equation";
import { formatShare } from "../format";

function ShareSlider() {
  const [value, setValue] = useState(0.3);
  return (
    <div className="max-w-sm">
      <Slider
        label="Share of eligible clients who move"
        value={value}
        min={0}
        max={1}
        step={0.05}
        display={formatShare(value, 0)}
        onChange={setValue}
      />
    </div>
  );
}

const meta = {
  title: "Controls/Slider and equation",
  component: ShareSlider,
} satisfies Meta<typeof ShareSlider>;

export default meta;
type Story = StoryObj<typeof meta>;

export const SliderControl: Story = {};

export const DisplayEquation: Story = {
  render: () => (
    <Equation label="Dirichlet posterior" tex={String.raw`P_{i\cdot} \mid N \sim \operatorname{Dirichlet}(1 + N_{i1}, \dots, 1 + N_{iK})`} />
  ),
};
