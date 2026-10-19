import type { Meta, StoryObj } from "@storybook/react";

// The design tokens as they render: every swatch is a CSS variable from src/styles/tokens.css, so
// the light and dark halves show the two sets of values for the same names.

function Swatch({ name }: { name: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <span className="h-10 rounded-control border border-border" style={{ background: `var(--${name})` }} />
      <span className="truncate font-mono text-label text-ink-secondary">{name}</span>
    </div>
  );
}

function Group({ title, names }: { title: string; names: string[] }) {
  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-small font-title text-ink">{title}</h2>
      <div className="grid grid-cols-3 gap-3 sm:grid-cols-5">
        {names.map((name) => (
          <Swatch key={name} name={name} />
        ))}
      </div>
    </section>
  );
}

function Foundations() {
  return (
    <div className="flex flex-col gap-6">
      <Group title="Surfaces and ink" names={["canvas", "surface", "surface-raised", "ink", "ink-secondary"]} />
      <Group title="Neutral ramp" names={["grey-1", "grey-2", "grey-3", "grey-4", "grey-5", "grey-6", "grey-7", "grey-8", "grey-9"]} />
      <Group title="Accent and status" names={["accent", "accent-fill", "accent-tint", "success", "warning", "danger"]} />
      <Group title="Robustness verdicts" names={["verdict-holds", "verdict-weakens", "verdict-reverses"]} />
      <Group title="Categorical series (equal lightness, fixed order)" names={["chart-1", "chart-2", "chart-3", "chart-4", "chart-5"]} />
      <Group title="Sequential and diverging ends" names={["chart-seq-low", "chart-seq-high", "chart-div-low", "chart-div-high"]} />
      <section className="flex flex-col gap-2">
        <h2 className="text-small font-title text-ink">Type</h2>
        <p className="text-title font-title text-ink">Title, tracked tight</p>
        <p className="text-heading font-title text-ink">Heading</p>
        <p className="text-body text-ink">Body text at the reading measure, with tabular figures: 1,234.5 and 6,789.0.</p>
        <p className="text-small text-ink-secondary">Small text for captions and dense tables.</p>
        <p className="text-label font-label text-ink-secondary">Label, tracked open</p>
      </section>
    </div>
  );
}

const meta = {
  title: "Foundations/Tokens",
  component: Foundations,
} satisfies Meta<typeof Foundations>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Tokens: Story = {};
