import type { Meta, StoryObj } from "@storybook/react";

import { FindingCard, VerdictCard } from "../components/Findings";
import { claims } from "../test-fixtures";

const meta = {
  title: "Findings/Claim cards",
  component: FindingCard,
  args: { claim: claims.claims[0]! },
  decorators: [
    (Story) => (
      <div className="max-w-xl">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof FindingCard>;

export default meta;
type Story = StoryObj<typeof meta>;

/** A published claim: the question as the title, values with intervals, the verdict and its reason, n and period. */
export const Published: Story = {};

/** A claim the checker has not judged carries no badge; its method takes the reason's place. */
export const NotJudged: Story = { args: { claim: { ...claims.claims[0]!, verdict: null, mechanism: null } } };

/** A verdict-only headline: words, stratifier names and the mechanism; never a number. */
export const VerdictOnly: Story = {
  render: () => <VerdictCard verdict={claims.verdicts[0]!} />,
};
