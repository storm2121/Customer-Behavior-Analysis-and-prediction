// The motion vocabulary (docs/DESIGN.md, section 3). Springs, not durations, for anything a person
// touches: they start from the value on screen and carry its velocity, so every transition can be
// interrupted and reversed. Bounce appears only after a drag release, when the gesture itself
// carried momentum. Under reduced motion every spring and slide becomes the same short cross-fade.

import { useReducedMotion } from "motion/react";

/** Critically damped: no overshoot, settles in about a third of a second. The default. */
export const SPRING = { type: "spring", bounce: 0, duration: 0.35 } as const;

/** After a drag release only: the thumb keeps the finger's velocity and settles with a little give. */
export const RELEASE_SPRING = { type: "spring", bounce: 0.2, duration: 0.45 } as const;

/** What every spring and slide becomes when the person asks for reduced motion. */
export const CROSS_FADE = { duration: 0.2, ease: "linear" } as const;

/** How far a control sinks under the finger, on pointer-down. */
export const PRESS_SCALE = 0.97;

/** Whether the person asked for reduced motion (false until the preference is known). */
export function useCalmMotion(): boolean {
  return useReducedMotion() ?? false;
}

/** The current value of `prefers-reduced-motion`, for code outside React (the chart wrapper). */
export function prefersReducedMotion(): boolean {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}
