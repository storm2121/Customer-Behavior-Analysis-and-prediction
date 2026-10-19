import type { Meta, StoryObj } from "@storybook/react";

import { intervalSeries } from "../charts/marks";
import { Chart, type ChartOption } from "../components/Chart";
import { Panel } from "../components/ui";

// Illustrative values only: real ones come from the API at run time.
const bars: ChartOption = {
  tooltip: { trigger: "item" },
  xAxis: { type: "value", min: 0 },
  yAxis: { type: "category", inverse: true, data: ["Mixed use", "Digital only", "Branch only", "No activity"], axisLine: { show: false } },
  series: [
    {
      type: "bar",
      id: "share",
      barWidth: "45%",
      data: [
        { value: 0.5, itemStyle: { color: "token:state-mixed", borderRadius: [0, 4, 4, 0] } },
        { value: 0.3, itemStyle: { color: "token:state-digital-only", borderRadius: [0, 4, 4, 0] } },
        { value: 0.15, itemStyle: { color: "token:state-branch-only", borderRadius: [0, 4, 4, 0] } },
        { value: 0.05, itemStyle: { color: "token:state-none", borderRadius: [0, 4, 4, 0] } },
      ],
    },
    intervalSeries({
      id: "share-interval",
      intervals: [
        [0.47, 0.53],
        [0.27, 0.33],
        [0.13, 0.17],
        [0.04, 0.07],
      ],
    }),
  ],
} as ChartOption;

const fan: ChartOption = {
  tooltip: { trigger: "axis" },
  legend: { top: 0, left: 0, data: ["With the policy", "Without it"] },
  xAxis: { type: "category", data: ["0", "1", "2", "3", "4", "5"], boundaryGap: false },
  yAxis: { type: "value", scale: true },
  series: [
    { type: "line", id: "baseline-low", data: [0.5, 0.49, 0.48, 0.48, 0.47, 0.47], stack: "baseline", symbol: "none", lineStyle: { opacity: 0 }, silent: true },
    { type: "line", id: "baseline-band", data: [0, 0.02, 0.04, 0.05, 0.06, 0.06], stack: "baseline", symbol: "none", lineStyle: { opacity: 0 }, areaStyle: { color: "token:chart-neutral", opacity: 0.14 }, silent: true },
    { type: "line", id: "baseline", name: "Without it", data: [0.5, 0.5, 0.5, 0.505, 0.5, 0.5], symbol: "none", lineStyle: { color: "token:chart-neutral" }, itemStyle: { color: "token:chart-neutral" } },
    { type: "line", id: "policy-low", data: [0.5, 0.53, 0.55, 0.56, 0.56, 0.57], stack: "policy", symbol: "none", lineStyle: { opacity: 0 }, silent: true },
    { type: "line", id: "policy-band", data: [0, 0.02, 0.04, 0.05, 0.06, 0.06], stack: "policy", symbol: "none", lineStyle: { opacity: 0 }, areaStyle: { color: "token:accent", opacity: 0.14 }, silent: true },
    { type: "line", id: "policy", name: "With the policy", data: [0.5, 0.54, 0.57, 0.585, 0.59, 0.6], symbol: "none", lineStyle: { color: "token:accent" }, itemStyle: { color: "token:accent" } },
  ],
} as ChartOption;

const meta = {
  title: "Charts/Chart",
  component: Chart,
  args: { label: "Bar chart of illustrative shares by channel profile, with intervals", height: "compact", option: bars },
  decorators: [
    (Story) => (
      <Panel title="Illustrative chart" subtitle="Values are made up; real ones come from the API.">
        <Story />
      </Panel>
    ),
  ],
} satisfies Meta<typeof Chart>;

export default meta;
type Story = StoryObj<typeof meta>;

/** One series in the channel states' colours, each bar with its interval whisker. */
export const BarsWithIntervals: Story = {};

/** The simulator's fan: the policy in the accent, the baseline in the neutral ramp, each with its band. */
export const Bands: Story = { args: { label: "Line chart of an illustrative policy against its baseline, with bands", height: "standard", option: fan } };
