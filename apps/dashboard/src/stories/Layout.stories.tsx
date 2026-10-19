import type { Meta, StoryObj } from "@storybook/react";

import { Figure, Panel, Tag } from "../components/ui";

const meta = {
  title: "Layout/Figure and panel",
  component: Figure,
  args: { label: "Mixed use", value: "50.0%", interval: "(47.0 to 53.0)", context: "of clients used the branch and a digital channel during the year" },
} satisfies Meta<typeof Figure>;

export default meta;
type Story = StoryObj<typeof meta>;

/** A number with its interval and one line of context: a line of a column, not a tile. Illustrative values. */
export const NumberWithInterval: Story = {
  render: (args) => (
    <dl className="max-w-xs">
      <Figure {...args} />
    </dl>
  ),
};

/** The overview's column: numbers read top to bottom, each carrying its interval or its sample. */
export const Column: Story = {
  render: () => (
    <dl className="flex max-w-xs flex-col gap-6">
      <Figure label="Clients" value="1,000" context="synthetic clients in the client file" />
      <Figure label="Mixed use" value="50.0%" interval="(47.0 to 53.0)" context="of clients used the branch and a digital channel" />
      <Figure label="Branch only" value="40.0%" interval="(37.0 to 43.0)" context="used the branch and no digital channel" />
    </dl>
  ),
};

/** An in-flow panel: a hairline border on the surface and no shadow. */
export const InFlowPanel: Story = {
  render: () => (
    <Panel title="Profiles" subtitle="An in-flow panel: hairline border, panel radius, no shadow.">
      <div className="flex flex-wrap gap-2">
        <Tag>Eligible age bands: under 25, 25 to 34</Tag>
        <Tag>A what-if under the fitted chain, not a forecast</Tag>
      </div>
    </Panel>
  ),
};
