import type { Bands } from "./api";
import { bandsAt, computedPolicies } from "./simulator";

const bands = (value: number): Bands => ({ digital_share: { p05: [value - 0.1], p50: [value], p95: [value + 0.1] } });

describe("the simulator's interpolation", () => {
  const points = computedPolicies([
    { share: 1, policy: bands(0.8), baseline: bands(0.5) },
    { share: 0.3, policy: bands(0.6), baseline: bands(0.5) },
  ]);

  it("takes each answer's policy at its share and the baseline at zero", () => {
    expect(points.map((point) => point.share).sort()).toEqual([0, 0.3, 1]);
  });

  it("is exact at a computed share", () => {
    expect(bandsAt(points, 0.3)?.digital_share?.p50).toEqual([0.6]);
    expect(bandsAt(points, 0)?.digital_share?.p50).toEqual([0.5]);
  });

  it("interpolates linearly between the two nearest computed shares, every percentile alike", () => {
    const mid = bandsAt(points, 0.65)?.digital_share;
    expect(mid?.p50[0]).toBeCloseTo(0.7);
    expect(mid?.p05[0]).toBeCloseTo(0.6);
    expect(mid?.p95[0]).toBeCloseTo(0.8);
    expect(bandsAt(points, 0.15)?.digital_share?.p50[0]).toBeCloseTo(0.55);
  });

  it("does not extrapolate, and has nothing to say before any answer", () => {
    expect(bandsAt(points, 1.5)?.digital_share?.p50).toEqual([0.8]);
    expect(bandsAt([], 0.3)).toBeNull();
  });
});
