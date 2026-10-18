import type { Meta, StoryObj } from "@storybook/react";

import { Kpi } from "../components/ui";

const meta = {
  title: "Layout/KPI tile",
  component: Kpi,
  args: { label: "Clients", value: "1,000", hint: "illustrative value" },
} satisfies Meta<typeof Kpi>;

export default meta;
type Story = StoryObj<typeof meta>;

export const WithHint: Story = {};
export const WithoutHint: Story = { args: { hint: undefined } };
