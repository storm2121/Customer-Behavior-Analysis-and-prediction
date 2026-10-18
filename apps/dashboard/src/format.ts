// Formatting shared by every page, so a share or an interval looks the same everywhere.

const integer = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });

export function formatCount(value: number): string {
  return integer.format(value);
}

export function formatShare(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "not defined";
  return `${(100 * value).toFixed(digits)}%`;
}

export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "not defined";
  return value.toLocaleString("en-GB", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** A value in the unit the ledger gives it; unknown units are shown as plain numbers. */
export function formatValue(value: number, unit?: string): string {
  switch (unit) {
    case "share":
      return formatShare(value);
    case "difference":
      return `${value >= 0 ? "+" : ""}${(100 * value).toFixed(1)} points`;
    case "count":
      return formatCount(value);
    case "ratio":
      return `${value.toFixed(2)}×`;
    case "days":
      return `${value.toFixed(0)} days`;
    case "p_value":
      return value < 0.001 ? "below 0.001" : value.toFixed(3);
    default:
      return formatNumber(value, Math.abs(value) >= 100 ? 0 : 3);
  }
}

export function formatInterval(
  interval: [number | null, number | null] | null | undefined,
  unit?: string,
): string {
  if (!interval) return "";
  const [low, high] = interval;
  if (low === null || high === null) return "";
  return `${formatValue(low, unit)} to ${formatValue(high, unit)}`;
}

export function humanize(key: string): string {
  return key.replaceAll("_", " ");
}
