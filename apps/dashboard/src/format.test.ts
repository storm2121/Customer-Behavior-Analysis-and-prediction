import { formatCount, formatInterval, formatNumber, formatShare, formatValue, humanize } from "./format";

describe("formatting", () => {
  it("formats shares, counts and plain numbers", () => {
    expect(formatShare(0.123)).toBe("12.3%");
    expect(formatShare(0.5, 0)).toBe("50%");
    expect(formatShare(null)).toBe("not defined");
    expect(formatShare(Number.NaN)).toBe("not defined");
    expect(formatCount(1234)).toBe("1,234");
    expect(formatNumber(0.5, 2)).toBe("0.50");
  });

  it("formats a value in the unit the ledger gives it", () => {
    expect(formatValue(0.25, "share")).toBe("25.0%");
    expect(formatValue(0.05, "difference")).toBe("+5.0 points");
    expect(formatValue(-0.05, "difference")).toBe("-5.0 points");
    expect(formatValue(1234, "count")).toBe("1,234");
    expect(formatValue(1.5, "ratio")).toBe("1.50×");
    expect(formatValue(30, "days")).toBe("30 days");
    expect(formatValue(0.0001, "p_value")).toBe("below 0.001");
    expect(formatValue(0.25)).toBe("0.250");
  });

  it("formats intervals and leaves undefined ends out", () => {
    expect(formatInterval([0.1, 0.2], "share")).toBe("10.0% to 20.0%");
    expect(formatInterval([null, 0.2], "share")).toBe("");
    expect(formatInterval(null)).toBe("");
  });

  it("turns keys into words", () => {
    expect(humanize("branch_only")).toBe("branch only");
  });
});
