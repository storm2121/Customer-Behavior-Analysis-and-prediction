import type { LucideIcon } from "lucide-react";

/**
 * A Lucide icon at one of the two sizes the dashboard uses, with the house stroke. An icon is
 * decorative (hidden from assistive technology) unless it is the only thing that names its
 * control, in which case the control carries the accessible name, never the icon alone.
 */
export function Icon({ icon: Glyph, size = 16 }: { icon: LucideIcon; size?: 16 | 20 }) {
  return <Glyph size={size} strokeWidth={1.75} aria-hidden focusable={false} className="shrink-0" />;
}
