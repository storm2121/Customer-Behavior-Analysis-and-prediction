import type { Meta, StoryObj } from "@storybook/react";

import { FindingCard, VerdictCard } from "../components/Findings";
import { claims } from "../test-fixtures";

const meta = {
  title: "Findings/Claim cards",
  component: FindingCard,
  args: { claim: claims.claims[0]! },
} satisfies Meta<typeof FindingCard>;

export default meta;
type Story = StoryObj<typeof meta>;

/** A published claim: values with intervals, the clients behind them, the checker's verdict. */
export const Published: Story = {};

/** A claim the checker has not judged carries no badge. */
export const NotJudged: Story = { args: { claim: { ...claims.claims[0]!, verdict: null } } };

/** A verdict-only headline: words, stratifier names and the mechanism; never a number. */
export const VerdictOnly: Story = {
  render: () => <VerdictCard verdict={claims.verdicts[0]!} />,
};
