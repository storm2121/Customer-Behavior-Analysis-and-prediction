// The four monthly channel states, with one identity each on every page: the same word and the
// same colour token wherever a state appears (the overview's mix, a client's timeline).

export type ChannelStateKey = "mixed" | "digital_only" | "branch_only" | "none";

export const CHANNEL_STATES: { key: ChannelStateKey; label: string; colour: string }[] = [
  { key: "mixed", label: "Mixed use", colour: "token:state-mixed" },
  { key: "digital_only", label: "Digital only", colour: "token:state-digital-only" },
  { key: "branch_only", label: "Branch only", colour: "token:state-branch-only" },
  { key: "none", label: "No activity", colour: "token:state-none" },
];

export function stateLabel(key: string): string {
  return CHANNEL_STATES.find((state) => state.key === key)?.label ?? key.replaceAll("_", " ");
}

export function stateColour(key: string): string {
  return CHANNEL_STATES.find((state) => state.key === key)?.colour ?? "token:chart-neutral";
}

/** A token name (`token:<name>`) as a CSS value, for swatches drawn in HTML beside a chart. */
export function cssToken(colour: string): string {
  return colour.startsWith("token:") ? `var(--${colour.slice("token:".length)})` : colour;
}
