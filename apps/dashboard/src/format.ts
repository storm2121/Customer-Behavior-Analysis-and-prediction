// Formatting shared by every page, so a share or an interval reads the same everywhere
// (docs/DESIGN.md, section 4): a value carries its unit, a percentage has one decimal at most, and
// an interval follows its value in the value's own unit and decimals, without repeating the unit:
// `48.4% (41.2 to 55.1)`.

const integer = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });

type Bounds = [number | null, number | null] | null | undefined;

const MINUS = "-";

function finite(value: number | null | undefined): value is number {
  return value !== null && value !== undefined && Number.isFinite(value);
}

export function formatCount(value: number): string {
  return integer.format(value);
}

export function formatShare(value: number | null | undefined, digits = 1): string {
  if (!finite(value)) return "not defined";
  return `${(100 * value).toFixed(digits)}%`;
}

export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (!finite(value)) return "not defined";
  return value.toLocaleString("en-GB", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function signed(value: number, digits: number): string {
  const text = Math.abs(value).toFixed(digits);
  if (Number(text) === 0) return text;
  return `${value > 0 ? "+" : MINUS}${text}`;
}

/** Decimals for a plain number, as the ledger's prose writes them: two significant figures under one. */
export function numberDigits(value: number): number {
  const magnitude = Math.abs(value);
  if (magnitude >= 1000) return 0;
  if (magnitude >= 10) return 1;
  if (magnitude >= 1 || magnitude === 0) return 2;
  return Math.max(2, 1 - Math.floor(Math.log10(magnitude)));
}

/** A value in the unit the ledger gives it; an unknown unit is a plain number. */
export function formatValue(value: number, unit?: string): string {
  if (!Number.isFinite(value)) return "not defined";
  switch (unit) {
    case "share":
      return formatShare(value);
    case "difference":
      return `${signed(100 * value, 1)} points`;
    case "count":
      return formatCount(value);
    case "ratio":
      return `${value.toFixed(2)}×`;
    case "days":
      return `${formatCount(value)} days`;
    case "amount":
      return `${formatNumber(value, Math.abs(value) >= 10 ? 0 : 2)} MAD`;
    case "p_value":
      return value < 0.001 ? "below 0.001" : value.toFixed(3);
    default:
      return formatNumber(value, numberDigits(value));
  }
}

/** An interval bound: the value's unit and decimals, without the unit's symbol or word. */
function bound(value: number, unit: string | undefined, reference: number): string {
  switch (unit) {
    case "share":
      return (100 * value).toFixed(1);
    case "difference":
      return signed(100 * value, 1);
    case "count":
    case "days":
      return formatCount(value);
    case "ratio":
      return value.toFixed(2);
    case "amount":
      return formatNumber(value, Math.abs(reference) >= 10 ? 0 : 2);
    default:
      return formatNumber(value, numberDigits(reference));
  }
}

/** `(41.2 to 55.1)`, or nothing when an end is missing. `reference` sets the decimals. */
export function formatInterval(interval: Bounds, unit?: string, reference?: number): string {
  if (!interval) return "";
  const [low, high] = interval;
  if (!finite(low) || !finite(high)) return "";
  const basis = finite(reference) ? reference : (low + high) / 2;
  return `(${bound(low, unit, basis)} to ${bound(high, unit, basis)})`;
}

/** A value and its interval: `48.4% (41.2 to 55.1)`. */
export function formatEstimate(value: number, interval: Bounds, unit?: string): string {
  const text = formatValue(value, unit);
  const around = formatInterval(interval, unit, value);
  return around ? `${text} ${around}` : text;
}

/** A plain number to fixed decimals with its interval, for a table that sets one precision. */
export function formatFixed(value: number | null | undefined, interval: Bounds, digits: number, suffix = ""): string {
  if (!finite(value)) return "not defined";
  const text = `${formatNumber(value, digits)}${suffix}`;
  if (!interval || !finite(interval[0]) || !finite(interval[1])) return text;
  return `${text} (${formatNumber(interval[0], digits)} to ${formatNumber(interval[1], digits)})`;
}

const longDate = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
const shortMonth = new Intl.DateTimeFormat("en-GB", { month: "short", year: "numeric", timeZone: "UTC" });

function asDate(iso: string): Date {
  return new Date(iso.length === 10 ? `${iso}T00:00:00Z` : iso.endsWith("Z") ? iso : `${iso}Z`);
}

/** `2 October 2023`. */
export function formatDate(iso: string): string {
  return longDate.format(asDate(iso));
}

/** `Jan 2024`, for a month on an axis or in a table. */
export function formatMonth(iso: string): string {
  return shortMonth.format(asDate(iso.slice(0, 10)));
}

/** `2 October 2023 to 2 October 2024`. */
export function formatPeriod(period: [string, string] | string[] | null | undefined): string {
  if (!period || period.length < 2) return "";
  return `${formatDate(period[0] ?? "")} to ${formatDate(period[1] ?? "")}`;
}

export function humanize(key: string): string {
  return key.replaceAll("_", " ");
}

const WORDS: Record<string, string> = {
  ari: "ARI",
  auc: "AUC",
  bic: "BIC",
  chi2: "chi-square",
  mad: "MAD",
  mon: "Mon",
  tue: "Tue",
  wed: "Wed",
  thu: "Thu",
  fri: "Fri",
  sat: "Sat",
  sun: "Sun",
};

/**
 * A sentence-case label from a key: `share_digital_only` becomes `Share digital only`, an age band
 * `branch_only_25_34` becomes `Branch only 25 to 34`, and acronyms and weekdays keep their case.
 */
export function label(key: string): string {
  const words = key
    .split("_")
    .map((word) => WORDS[word] ?? word)
    .join(" ")
    .replace(/(\d+) (\d+)/g, "$1 to $2");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** A model feature in words: a count drops its `n_` prefix, so `n_events_90d` reads `Events 90d`. */
export function featureLabel(key: string): string {
  return label(key.replace(/^n_/, ""));
}
