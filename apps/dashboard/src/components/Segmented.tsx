import { motion } from "motion/react";
import { useId } from "react";

import { CROSS_FADE, SPRING, useCalmMotion } from "../motion";

/**
 * A segmented control: native radio buttons (arrow keys move the choice, the group has one tab
 * stop), drawn as one track with a raised segment that slides to the choice on a spring. The
 * slide starts from wherever the segment is, so a quick second choice redirects it mid-way.
 */
export function Segmented<T extends string | number>({
  label,
  options,
  value,
  onChange,
  hideLabel = false,
}: {
  label: string;
  options: { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
  hideLabel?: boolean;
}) {
  const id = useId();
  const calm = useCalmMotion();
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <span id={`${id}-label`} className={hideLabel ? "sr-only" : "text-small font-label text-ink"}>
        {label}
      </span>
      <div
        role="radiogroup"
        aria-labelledby={`${id}-label`}
        className="inline-flex w-fit max-w-full rounded-control bg-fill-control p-0.5"
      >
        {options.map((option) => {
          const selected = option.value === value;
          return (
            <label
              key={String(option.value)}
              className={`relative flex min-h-11 cursor-pointer select-none items-center justify-center rounded-control px-3 text-small sm:min-h-8 has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-1 has-[:focus-visible]:outline-accent has-[:focus-visible]:outline-solid ${
                selected ? "font-label text-ink" : "text-ink-secondary hover:text-ink"
              }`}
            >
              <input
                type="radio"
                name={id}
                className="sr-only"
                checked={selected}
                onChange={() => onChange(option.value)}
              />
              {selected ? (
                <motion.span
                  layoutId={`${id}-thumb`}
                  transition={calm ? CROSS_FADE : SPRING}
                  className="absolute inset-0 rounded-control border border-border bg-surface-raised shadow-thumb"
                  aria-hidden
                />
              ) : null}
              <span className="relative">{option.label}</span>
            </label>
          );
        })}
      </div>
    </div>
  );
}
