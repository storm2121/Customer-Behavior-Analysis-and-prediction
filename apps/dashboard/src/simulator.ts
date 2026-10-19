// The simulator's bands follow its sliders continuously while the network call waits for the
// slider to settle (docs/DESIGN.md, section 3: debounce the network, never the visual).
//
// Between policies the API has computed, the bands are interpolated linearly in the share moved.
// That is exact for each posterior draw: the projection x_h = x_0 P^h is linear in the starting
// counts, and moving a share s of the eligible clients makes x_0 linear in s, so every draw's
// digital share and branch load are linear in s. The quantiles across draws are close to linear
// but not exactly so; the bands of the value the slider settles on are always the API's own.

import type { Bands } from "./api";

export interface ComputedPolicy {
  share: number;
  bands: Bands;
}

function mixBands(low: Bands, high: Bands, t: number): Bands {
  const out: Bands = {};
  for (const [quantity, levels] of Object.entries(low)) {
    const other = high[quantity];
    if (!other) continue;
    const mix = (a: number[], b: number[]) => a.map((value, index) => value + t * ((b[index] ?? value) - value));
    out[quantity] = { p05: mix(levels.p05, other.p05), p50: mix(levels.p50, other.p50), p95: mix(levels.p95, other.p95) };
  }
  return out;
}

/**
 * The bands for `share`, from the computed policies around it: exact when one was computed at
 * that share, interpolated between the nearest two otherwise, and the nearest one outside them
 * (no extrapolation). `null` when nothing is computed yet.
 */
export function bandsAt(points: ComputedPolicy[], share: number): Bands | null {
  if (points.length === 0) return null;
  const sorted = [...points].sort((a, b) => a.share - b.share);
  const first = sorted[0]!;
  const last = sorted[sorted.length - 1]!;
  if (share <= first.share) return first.bands;
  if (share >= last.share) return last.bands;
  for (let index = 1; index < sorted.length; index += 1) {
    const high = sorted[index]!;
    const low = sorted[index - 1]!;
    if (share <= high.share) {
      const span = high.share - low.share;
      return span > 0 ? mixBands(low.bands, high.bands, (share - low.share) / span) : high.bands;
    }
  }
  return last.bands;
}

/** Computed policies of one age limit and target: each response's policy, and its baseline at zero. */
export function computedPolicies(responses: { share: number; policy: Bands; baseline: Bands }[]): ComputedPolicy[] {
  const byShare = new Map<number, Bands>();
  for (const response of responses) {
    byShare.set(response.share, response.policy);
    if (!byShare.has(0)) byShare.set(0, response.baseline);
  }
  return [...byShare.entries()].map(([share, bands]) => ({ share, bands }));
}
