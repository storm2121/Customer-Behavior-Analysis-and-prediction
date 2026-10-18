import * as RadixSlider from "@radix-ui/react-slider";

/** A labelled single-thumb slider (Radix), keyboard and screen-reader friendly. */
export function Slider({
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (value: number) => void;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-sm font-medium text-ink">{label}</span>
        <span className="text-sm tabular-nums text-ink-muted">{display}</span>
      </div>
      <RadixSlider.Root
        className="relative flex h-6 w-full touch-none select-none items-center"
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(values) => onChange(values[0] ?? value)}
      >
        <RadixSlider.Track className="relative h-1.5 grow rounded-full bg-surface-muted">
          <RadixSlider.Range className="absolute h-full rounded-full bg-accent" />
        </RadixSlider.Track>
        <RadixSlider.Thumb
          aria-label={label}
          aria-valuetext={display}
          className="block h-5 w-5 rounded-full border-2 border-accent bg-surface shadow focus-visible:outline-2 focus-visible:outline-accent"
        />
      </RadixSlider.Root>
    </div>
  );
}
