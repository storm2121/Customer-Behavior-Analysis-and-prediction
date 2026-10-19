import type { LucideIcon } from "lucide-react";
import { motion, type HTMLMotionProps } from "motion/react";
import { forwardRef, type ReactNode } from "react";

import { PRESS_SCALE, SPRING, useCalmMotion } from "../motion";
import { Icon } from "./Icon";

type Tone = "primary" | "secondary" | "quiet";

// Colour changes are instant (no transition): the press shows the moment the pointer goes down,
// and only the scale moves, on a spring that any new press interrupts.
const TONES: Record<Tone, string> = {
  primary: "bg-accent-fill text-on-accent active:brightness-90",
  secondary: "border border-border bg-surface text-ink hover:bg-fill-subtle active:bg-fill-control",
  quiet: "text-ink-secondary hover:bg-fill-subtle hover:text-ink active:bg-fill-control",
};

const SHAPE =
  "inline-flex select-none items-center justify-center gap-2 rounded-control px-3 text-small font-label " +
  "min-h-11 sm:min-h-9 disabled:cursor-not-allowed disabled:opacity-60 [-webkit-tap-highlight-color:transparent]";

export interface ButtonProps extends Omit<HTMLMotionProps<"button">, "children"> {
  tone?: Tone;
  icon?: LucideIcon;
  children?: ReactNode;
}

/** A button that presses on pointer-down and springs back, interruptibly. */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { tone = "secondary", icon, children, className = "", type = "button", ...rest },
  ref,
) {
  const calm = useCalmMotion();
  return (
    <motion.button
      ref={ref}
      type={type}
      whileTap={calm || rest.disabled ? undefined : { scale: PRESS_SCALE }}
      transition={SPRING}
      className={`${SHAPE} ${TONES[tone]} ${className}`}
      {...rest}
    >
      {icon ? <Icon icon={icon} /> : null}
      {children}
    </motion.button>
  );
});

/** An icon-only button: the label is its accessible name, and shows as a native tooltip. */
export const IconButton = forwardRef<HTMLButtonElement, Omit<ButtonProps, "children"> & { label: string; icon: LucideIcon }>(
  function IconButton({ label, icon, tone = "quiet", className = "", ...rest }, ref) {
    const calm = useCalmMotion();
    return (
      <motion.button
        ref={ref}
        type="button"
        aria-label={label}
        title={label}
        whileTap={calm ? undefined : { scale: PRESS_SCALE }}
        transition={SPRING}
        className={`${SHAPE} min-w-11 px-0 sm:min-w-9 ${TONES[tone]} ${className}`}
        {...rest}
      >
        <Icon icon={icon} size={20} />
      </motion.button>
    );
  },
);
