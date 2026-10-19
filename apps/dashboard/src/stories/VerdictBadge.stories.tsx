import type { Meta, StoryObj } from "@storybook/react";

import { VerdictBadge } from "../components/ui";

const meta = {
  title: "Findings/Robustness badge",
  component: VerdictBadge,
  argTypes: { verdict: { control: "select", options: ["holds", "weakens", "reverses", "insufficient"] } },
} satisfies Meta<typeof VerdictBadge>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Holds: Story = { args: { verdict: "holds" } };
export const Weakens: Story = { args: { verdict: "weakens" } };
export const Reverses: Story = { args: { verdict: "reverses" } };
export const Insufficient: Story = { args: { verdict: "insufficient" } };

/** The three verdicts and the neutral one together: colour, icon and word, so none relies on colour. */
export const AllFour: Story = {
  args: { verdict: "holds" },
  render: () => (
    <div className="flex flex-wrap gap-2">
      {["holds", "weakens", "reverses", "insufficient"].map((verdict) => (
        <VerdictBadge key={verdict} verdict={verdict} />
      ))}
    </div>
  ),
};
