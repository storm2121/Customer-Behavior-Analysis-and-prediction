import { animate, motion, useMotionValue, useTransform, type AnimationPlaybackControls } from "motion/react";
import { useCallback, useEffect, useId, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";

import { RELEASE_SPRING, SPRING, useCalmMotion } from "../motion";

// The simulator's sliders are the dashboard's showcase of direct manipulation (docs/DESIGN.md,
// section 3): the thumb follows the pointer one to one under pointer capture, keeping the offset
// at which it was grabbed; past either end it resists progressively instead of stopping dead; on
// release the pointer's velocity becomes the spring's, so there is no seam between dragging and
// settling; and whatever the thumb is doing, a new touch or key press takes it from where it is.

/** Apple's rubber band: the further past the bound, the less the thumb follows (never past `limit`). */
export function rubberBand(overshoot: number, limit: number, constant = 0.55): number {
  if (limit <= 0) return 0;
  return (overshoot * limit * constant) / (limit + constant * Math.abs(overshoot));
}

/** `value` clamped to the range and rounded to the nearest step, without floating-point dust. */
export function snapToStep(value: number, min: number, max: number, step: number): number {
  const decimals = (String(step).split(".")[1] ?? "").length;
  const steps = Math.round((Math.min(max, Math.max(min, value)) - min) / step);
  return Number(Math.min(max, Math.max(min, min + steps * step)).toFixed(decimals));
}

/** The value a key press asks for: one step, ten with Shift or Page keys, or an end. */
export function keyedValue(key: string, shift: boolean, value: number, min: number, max: number, step: number): number | null {
  const steps = shift ? 10 : 1;
  switch (key) {
    case "ArrowRight":
    case "ArrowUp":
      return snapToStep(value + steps * step, min, max, step);
    case "ArrowLeft":
    case "ArrowDown":
      return snapToStep(value - steps * step, min, max, step);
    case "PageUp":
      return snapToStep(value + 10 * step, min, max, step);
    case "PageDown":
      return snapToStep(value - 10 * step, min, max, step);
    case "Home":
      return min;
    case "End":
      return max;
    default:
      return null;
  }
}

export interface SliderProps {
  label: string;
  /** The value in effect, always on a step. */
  value: number;
  min: number;
  max: number;
  step: number;
  /** How the value reads, on screen and to assistive technology. */
  format: (value: number) => string;
  /** Called whenever the stepped value changes: while dragging, on a key, on release. */
  onChange: (value: number) => void;
  /** The continuous value while the thumb moves (dragging, springing back); null once at rest. */
  onLive?: (value: number | null) => void;
  /** The first touch or focus, so the page can fetch what a change would need. */
  onIntent?: () => void;
}

export function Slider({ label, value, min, max, step, format, onChange, onLive, onIntent }: SliderProps) {
  const labelId = useId();
  const calm = useCalmMotion();
  const root = useRef<HTMLDivElement>(null);
  const track = useRef<HTMLDivElement>(null);
  const thumb = useRef<HTMLDivElement>(null);
  const width = useRef(0);
  const x = useMotionValue(0);
  const fill = useTransform(x, (position) => (width.current > 0 ? Math.min(1, Math.max(0, position / width.current)) : 0));
  const motionRef = useRef<AnimationPlaybackControls | null>(null);
  const drag = useRef<{ pointer: number; grab: number; from: number; last: number } | null>(null);
  const intended = useRef(false);
  const current = useRef(value);
  current.current = value;
  const [pressed, setPressed] = useState(false);

  const span = max - min;
  const toX = useCallback((v: number) => (span > 0 ? ((v - min) / span) * width.current : 0), [min, span]);
  const toValue = useCallback(
    (position: number) => (width.current > 0 ? min + (Math.min(width.current, Math.max(0, position)) / width.current) * span : min),
    [min, span],
  );

  const intent = () => {
    if (intended.current) return;
    intended.current = true;
    onIntent?.();
  };

  /** Move the thumb to `target` on a spring that starts from the thumb's position and speed. */
  const settle = useCallback(
    (target: number, velocity: number, spring: typeof SPRING | typeof RELEASE_SPRING) => {
      motionRef.current?.stop();
      const to = toX(target);
      if (calm) {
        motionRef.current = null;
        x.set(to);
        onLive?.(null);
        return;
      }
      const controls = animate(x, to, {
        ...spring,
        velocity,
        onUpdate: (latest) => onLive?.(toValue(latest)),
        onComplete: () => {
          if (motionRef.current === controls) motionRef.current = null;
          onLive?.(null);
        },
      });
      motionRef.current = controls;
    },
    [calm, onLive, toValue, toX, x],
  );

  // Keep the thumb on the value when the track is measured or resized, or the value changes from
  // outside a drag (a key press animates itself; a new value from the page springs to its place).
  useEffect(() => {
    const element = track.current;
    if (!element) return;
    const measure = () => {
      width.current = element.getBoundingClientRect().width;
      if (!drag.current) {
        motionRef.current?.stop();
        motionRef.current = null;
        x.set(toX(current.current));
      }
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [toX, x]);

  useEffect(() => {
    if (drag.current) return;
    if (Math.abs(toX(value) - x.get()) > 0.5 && !motionRef.current) settle(value, x.getVelocity(), SPRING);
  }, [settle, toX, value, x]);

  useEffect(() => () => motionRef.current?.stop(), []);

  const follow = (clientX: number) => {
    const state = drag.current;
    const element = track.current;
    if (!state || !element) return;
    const raw = clientX - element.getBoundingClientRect().left - state.grab;
    const limit = thumb.current?.offsetWidth ?? 0;
    const shown = raw < 0 ? -rubberBand(-raw, limit) : raw > width.current ? width.current + rubberBand(raw - width.current, limit) : raw;
    x.set(shown);
    const continuous = toValue(raw);
    onLive?.(continuous);
    const stepped = snapToStep(continuous, min, max, step);
    if (stepped !== state.last) {
      state.last = stepped;
      onChange(stepped);
    }
  };

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || !track.current || !thumb.current) return;
    intent();
    motionRef.current?.stop();
    motionRef.current = null;
    width.current = track.current.getBoundingClientRect().width;
    const left = track.current.getBoundingClientRect().left;
    const box = thumb.current.getBoundingClientRect();
    // Grabbing the thumb keeps the offset at which it was taken; pressing the track brings the
    // thumb under the pointer at once (the press responds on pointer-down, not on release).
    const onThumb = event.clientX >= box.left && event.clientX <= box.right;
    const grab = onThumb ? event.clientX - left - x.get() : 0;
    drag.current = { pointer: event.pointerId, grab, from: value, last: value };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    setPressed(true);
    thumb.current.focus({ preventScroll: true });
    follow(event.clientX);
    event.preventDefault();
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (drag.current?.pointer === event.pointerId) follow(event.clientX);
  };

  const finish = (event: PointerEvent<HTMLDivElement>, keep: boolean) => {
    const state = drag.current;
    if (!state || state.pointer !== event.pointerId) return;
    drag.current = null;
    setPressed(false);
    event.currentTarget.releasePointerCapture?.(event.pointerId);
    // A cancelled gesture (the browser took it to scroll the page) puts the value back.
    const target = keep ? state.last : state.from;
    if (!keep && state.last !== state.from) onChange(state.from);
    settle(target, x.getVelocity(), RELEASE_SPRING);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const next = keyedValue(event.key, event.shiftKey, value, min, max, step);
    if (next === null) return;
    event.preventDefault();
    intent();
    if (next !== value) onChange(next);
    settle(next, x.getVelocity(), SPRING);
  };

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between gap-3">
        <span id={labelId} className="text-small font-label text-ink">
          {label}
        </span>
        <span className="text-small text-ink-secondary" aria-hidden>
          {format(value)}
        </span>
      </div>
      <div
        ref={root}
        className="relative flex h-11 cursor-pointer touch-pan-y select-none items-center [-webkit-tap-highlight-color:transparent]"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={(event) => finish(event, true)}
        onPointerCancel={(event) => finish(event, false)}
        onLostPointerCapture={(event) => finish(event, true)}
      >
        <div ref={track} className="relative h-1 w-full rounded-full bg-fill-control">
          <motion.div className="absolute inset-0 origin-left rounded-full bg-accent" style={{ scaleX: fill }} />
          <motion.div
            ref={thumb}
            role="slider"
            tabIndex={0}
            aria-labelledby={labelId}
            aria-valuemin={min}
            aria-valuemax={max}
            aria-valuenow={value}
            aria-valuetext={format(value)}
            aria-orientation="horizontal"
            onKeyDown={onKeyDown}
            onFocus={intent}
            className="absolute top-1/2 left-0 -mt-3.5 -ml-3.5 size-7 rounded-full border border-border bg-surface-raised shadow-thumb"
            style={{ x }}
            animate={{ scale: pressed && !calm ? 1.12 : 1 }}
            transition={SPRING}
          />
        </div>
      </div>
    </div>
  );
}
