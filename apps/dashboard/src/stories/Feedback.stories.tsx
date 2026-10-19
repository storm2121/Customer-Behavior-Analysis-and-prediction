import type { Meta, StoryObj } from "@storybook/react";

import { ApiError } from "../api";
import { Bone, EmptyState, Failure, Loading, Notice } from "../components/ui";

const meta = {
  title: "Feedback/Notices",
  component: Notice,
  args: { kind: "status", children: "These claims are computed on synthetic data, like every other page." },
  argTypes: { kind: { control: "select", options: ["status", "completion", "warning", "error"] } },
} satisfies Meta<typeof Notice>;

export default meta;
type Story = StoryObj<typeof meta>;

/** The four kinds side by side: each has its own colour, icon and wording. */
export const FourKinds: Story = {
  render: () => (
    <div className="flex max-w-xl flex-col gap-3">
      <Notice kind="status">While a slider moves, the bands are interpolated between computed policies.</Notice>
      <Notice kind="completion">Dismissed. The review is stored with the alert.</Notice>
      <Notice kind="warning" title="Intervals are missing">
        The synthetic ledger is older than the warehouse, so the overview shows shares without intervals.
      </Notice>
      <Notice kind="error" title="This part of the page could not load">
        The API answered 503: not available yet.
      </Notice>
    </div>
  ),
};

/** An error says what happened and what to do, with the one command and a retry. */
export const ErrorWithAction: Story = {
  render: () => (
    <div className="max-w-xl">
      <Failure error={new ApiError(503, "not available yet: models/silence/results.json")} onRetry={() => undefined} />
    </div>
  ),
};

/** An empty state says what will appear, how to make it appear, and the command that does it. */
export const Empty: Story = {
  render: () => (
    <div className="max-w-xl">
      <EmptyState title="The queue is empty." command="just anomaly">
        Ranked alerts appear here once the monitoring layer has run on this workspace.
      </EmptyState>
    </div>
  ),
};

/** Loading is the content's shape, announced once, never a spinner. */
export const Skeleton: Story = {
  render: () => (
    <Loading what="the queue">
      {[0, 1, 2].map((index) => (
        <Bone key={index} className="h-14 w-full" />
      ))}
    </Loading>
  ),
};
