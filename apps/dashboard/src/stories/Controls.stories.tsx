import type { Meta, StoryObj } from "@storybook/react";
import { Search } from "lucide-react";
import { useState } from "react";

import { Equation } from "../components/Equation";
import { Button, IconButton } from "../components/Pressable";
import { Segmented } from "../components/Segmented";
import { Slider } from "../components/Slider";
import { formatShare } from "../format";

/** The simulator's slider: drag it past an end, release it mid-flick, or use the arrow keys (Shift for ten steps). */
function ShareSlider() {
  const [value, setValue] = useState(0.3);
  const [live, setLive] = useState<number | null>(null);
  return (
    <div className="flex max-w-sm flex-col gap-2">
      <Slider label="Share of eligible clients who move" value={value} min={0} max={1} step={0.05} format={(v) => formatShare(v, 0)} onChange={setValue} onLive={setLive} />
      <p className="text-label text-ink-secondary">{live === null ? "At rest" : `Following the thumb: ${formatShare(live)}`}</p>
    </div>
  );
}

function AgeSlider() {
  const [value, setValue] = useState(45);
  return (
    <div className="max-w-sm">
      <Slider label="Clients younger than" value={value} min={25} max={65} step={10} format={(v) => `${v} years`} onChange={setValue} />
    </div>
  );
}

function Target() {
  const [value, setValue] = useState<"mixed" | "digital_only">("mixed");
  return (
    <Segmented
      label="They move to"
      options={[
        { value: "mixed", label: "Mixed use" },
        { value: "digital_only", label: "Digital only" },
      ]}
      value={value}
      onChange={setValue}
    />
  );
}

const meta = {
  title: "Controls/Slider",
  component: ShareSlider,
} satisfies Meta<typeof ShareSlider>;

export default meta;
type Story = StoryObj<typeof meta>;

export const SliderShowcase: Story = {};

/** Steps of ten years: the thumb tracks the pointer, then settles on a step. */
export const SteppedSlider: Story = { render: () => <AgeSlider /> };

/** The selection slides to the choice on a spring and can be redirected half-way. */
export const SegmentedControl: Story = { render: () => <Target /> };

/** Buttons press on pointer-down; colour changes are instant, only the scale springs. */
export const Buttons: Story = {
  render: () => (
    <div className="flex flex-wrap items-center gap-3">
      <Button tone="primary" icon={Search}>
        Show
      </Button>
      <Button tone="secondary">Needs information</Button>
      <Button tone="quiet">Evidence</Button>
      <Button tone="secondary" disabled>
        Storing
      </Button>
      <IconButton label="Search" icon={Search} />
    </div>
  ),
};

export const DisplayEquation: Story = {
  render: () => (
    <div className="max-w-measure">
      <Equation label="Dirichlet posterior" tex={String.raw`P_{i\cdot} \mid N \sim \operatorname{Dirichlet}(1 + N_{i1}, \dots, 1 + N_{iK})`} />
    </div>
  ),
};
