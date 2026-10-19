import {
  formatCount,
  formatDate,
  formatEstimate,
  formatFixed,
  formatInterval,
  formatMonth,
  formatNumber,
  formatPeriod,
  formatShare,
  formatValue,
  featureLabel,
  humanize,
  label,
  numberDigits,
} from "./format";

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
    expect(formatValue(Number.NaN, "share")).toBe("not defined");
  });

  it("writes plain numbers with two significant figures under one, as the ledger does", () => {
    expect(numberDigits(0.0097)).toBe(4);
    expect(numberDigits(0.25)).toBe(2);
    expect(numberDigits(2.5)).toBe(2);
    expect(numberDigits(12.5)).toBe(1);
    expect(numberDigits(1234)).toBe(0);
    expect(formatValue(0.0097)).toBe("0.0097");
    expect(formatValue(0.25)).toBe("0.25");
  });

  it("follows a value with its interval in the value's unit and decimals, the unit said once", () => {
    expect(formatEstimate(0.484, [0.412, 0.551], "share")).toBe("48.4% (41.2 to 55.1)");
    expect(formatEstimate(0.05, [0.025, 0.074], "difference")).toBe("+5.0 points (+2.5 to +7.4)");
    expect(formatEstimate(2.5, [2.1, 2.9], "ratio")).toBe("2.50× (2.10 to 2.90)");
    expect(formatEstimate(0.0097, [0.0082, 0.013])).toBe("0.0097 (0.0082 to 0.0130)");
    expect(formatEstimate(0.25, null, "share")).toBe("25.0%");
    expect(formatInterval([0.1, 0.2], "share")).toBe("(10.0 to 20.0)");
    expect(formatInterval([null, 0.2], "share")).toBe("");
    expect(formatInterval(null)).toBe("");
  });

  it("keeps one precision across a table", () => {
    expect(formatFixed(0.7123, [0.68, 0.745], 3)).toBe("0.712 (0.680 to 0.745)");
    expect(formatFixed(2.5, [2.1, 2.9], 2, "×")).toBe("2.50× (2.10 to 2.90)");
    expect(formatFixed(null, null, 3)).toBe("not defined");
  });

  it("writes dates, months and periods in words", () => {
    expect(formatDate("2023-10-02")).toBe("2 October 2023");
    expect(formatMonth("2024-01-01T00:00:00.000")).toBe("Jan 2024");
    expect(formatPeriod(["2023-10-02", "2024-10-02"])).toBe("2 October 2023 to 2 October 2024");
    expect(formatPeriod(null)).toBe("");
  });

  it("turns keys into words", () => {
    expect(humanize("branch_only")).toBe("branch only");
    expect(label("share_digital_only")).toBe("Share digital only");
    expect(label("branch_only_25_34")).toBe("Branch only 25 to 34");
    expect(label("mean_ari")).toBe("Mean ARI");
    expect(label("card_fri")).toBe("Card Fri");
    expect(featureLabel("n_events_90d")).toBe("Events 90d");
  });
});
