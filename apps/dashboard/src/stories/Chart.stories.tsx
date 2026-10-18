import type { Meta, StoryObj } from "@storybook/react";

import { Chart } from "../components/Chart";
import { Card } from "../components/ui";

const meta = {
  title: "Charts/Chart",
  component: Chart,
  args: {
    label: "Bar chart of an illustrative share by channel profile",
    height: 280,
    option: {
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: ["branch only", "mixed", "digital only", "none"] },
      yAxis: { type: "value" },
      series: [{ type: "bar", data: [0.4, 0.3, 0.2, 0.1], barMaxWidth: 56 }],
    },
  },
  decorators: [
    (Story) => (
      <Card title="Illustrative chart" subtitle="Values are made up; real ones come from the API.">
        <Story />
      </Card>
    ),
  ],
} satisfies Meta<typeof Chart>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Bars: Story = {};

export const Lines: Story = {
  args: {
    label: "Line chart of two illustrative series",
    option: {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      xAxis: { type: "category", data: ["0", "1", "2", "3"] },
      yAxis: { type: "value" },
      series: [
        { type: "line", name: "baseline", data: [0.5, 0.52, 0.53, 0.55] },
        { type: "line", name: "policy", data: [0.5, 0.58, 0.61, 0.63] },
      ],
    },
  },
};
