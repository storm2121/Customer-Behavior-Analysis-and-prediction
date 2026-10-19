// The ECharts theme, generated from the design tokens (styles/tokens.css) at run time.
//
// ECharts 5 binds a registered theme when a chart is created, so switching it would mean
// disposing and re-creating every chart (a remount, with its entrance animation). Instead the
// theme is an object of component defaults built from the tokens in effect on the chart's own
// element, and merged into each option: a theme change is then an ordinary update that keeps the
// chart instance. Reading the tokens from the element (not the document) lets a themed subtree,
// such as Storybook's side-by-side view, draw its charts in its own theme.

import type { EChartsOption } from "echarts";

export type ChartOption = EChartsOption;

type Dict = Record<string, unknown>;

export type ChannelState = "digital_only" | "mixed" | "branch_only" | "none";

/** The token values a chart needs, read once per paint. */
export interface ChartTokens {
  font: string;
  /** The label size and the 4px base unit, in CSS pixels (a canvas wants numbers). */
  labelSize: number;
  unit: number;
  ink: string;
  label: string;
  axis: string;
  grid: string;
  surface: string;
  raised: string;
  border: string;
  accent: string;
  neutral: string;
  neutralSoft: string;
  series: string[];
  sequential: [string, string];
  diverging: [string, string];
  states: Record<ChannelState, string>;
  /** Chart durations in milliseconds: updates, and the entrance on a page's first paint. */
  update: number;
  enter: number;
  /** Any token's value, for options that name one as `token:<name>`. */
  resolve: (name: string) => string;
}

/** A colour named by its token, resolved on the chart's own element when the chart is drawn. */
export function tokenColour(name: string): string {
  return `token:${name}`;
}

/** Token names the chart theme reads; tokens.test.ts checks that each one is defined. */
export const CHART_TOKEN_NAMES = [
  "font-family-sans",
  "type-label-size",
  "space-unit",
  "chart-ink",
  "chart-label",
  "chart-axis",
  "chart-grid",
  "chart-surface",
  "surface-raised",
  "border",
  "accent",
  "chart-neutral",
  "chart-neutral-soft",
  "chart-1",
  "chart-2",
  "chart-3",
  "chart-4",
  "chart-5",
  "chart-seq-low",
  "chart-seq-high",
  "chart-div-low",
  "chart-div-high",
  "state-digital-only",
  "state-mixed",
  "state-branch-only",
  "state-none",
  "duration-chart",
  "duration-chart-enter",
] as const;

function pixels(value: string, rootSize: number): number {
  const number = Number.parseFloat(value);
  if (!Number.isFinite(number)) return 0;
  return value.trim().endsWith("rem") ? number * rootSize : number;
}

function milliseconds(value: string): number {
  const number = Number.parseFloat(value);
  if (!Number.isFinite(number)) return 0;
  return value.trim().endsWith("ms") ? number : number * 1000;
}

/** The chart tokens in effect on `element`, through `read` (a test can pass its own). */
export function readChartTokens(
  element: Element = document.documentElement,
  read: (name: string) => string = (name) => getComputedStyle(element).getPropertyValue(`--${name}`).trim(),
): ChartTokens {
  const rootSize = Number.parseFloat(getComputedStyle(document.documentElement).fontSize) || 0;
  return {
    resolve: read,
    font: read("font-family-sans"),
    labelSize: pixels(read("type-label-size"), rootSize),
    unit: pixels(read("space-unit"), rootSize),
    ink: read("chart-ink"),
    label: read("chart-label"),
    axis: read("chart-axis"),
    grid: read("chart-grid"),
    surface: read("chart-surface"),
    raised: read("surface-raised"),
    border: read("border"),
    accent: read("accent"),
    neutral: read("chart-neutral"),
    neutralSoft: read("chart-neutral-soft"),
    series: (["chart-1", "chart-2", "chart-3", "chart-4", "chart-5"] as const).map(read),
    sequential: [read("chart-seq-low"), read("chart-seq-high")],
    diverging: [read("chart-div-low"), read("chart-div-high")],
    states: {
      digital_only: read("state-digital-only"),
      mixed: read("state-mixed"),
      branch_only: read("state-branch-only"),
      none: read("state-none"),
    },
    update: milliseconds(read("duration-chart")),
    enter: milliseconds(read("duration-chart-enter")),
  };
}

/**
 * Component defaults in option form. Text wears the text tokens, never a series colour; grids and
 * axes are solid hairlines one step off the surface; ticks are off; the tooltip is a raised,
 * floating surface; categorical colours come in the fixed order of the tokens.
 */
export function chartTheme(t: ChartTokens): Dict {
  const text = { color: t.label, fontFamily: t.font, fontSize: t.labelSize };
  const axis = {
    axisLine: { lineStyle: { color: t.axis, width: 1 } },
    axisTick: { show: false },
    axisLabel: { ...text, margin: 2 * t.unit },
    nameTextStyle: text,
    splitLine: { lineStyle: { color: t.grid, width: 1 } },
  };
  return {
    color: t.series,
    backgroundColor: "transparent",
    textStyle: text,
    tooltip: {
      backgroundColor: t.raised,
      borderColor: t.border,
      borderWidth: 1,
      padding: [2 * t.unit, 3 * t.unit],
      textStyle: { color: t.ink, fontFamily: t.font, fontSize: t.labelSize },
      className: "chart-tooltip",
      extraCssText: "border-radius: var(--shape-control); box-shadow: var(--elevation-float);",
      confine: true,
      transitionDuration: 0,
      axisPointer: { lineStyle: { color: t.axis, width: 1 }, crossStyle: { color: t.axis } },
    },
    legend: {
      textStyle: text,
      inactiveColor: t.neutralSoft,
      itemWidth: 3 * t.unit,
      itemHeight: 2 * t.unit,
      itemGap: 4 * t.unit,
    },
    grid: { left: 2 * t.unit, right: 4 * t.unit, top: 8 * t.unit, bottom: 2 * t.unit, containLabel: true },
    xAxis: axis,
    yAxis: axis,
    visualMap: { textStyle: text },
    calendar: {
      dayLabel: text,
      monthLabel: text,
      yearLabel: { show: false },
      itemStyle: { color: t.surface, borderColor: t.surface, borderWidth: 2 },
      splitLine: { show: false },
    },
  };
}

/** Defaults by series type: two-pixel lines, eight-pixel markers, bars capped in thickness. */
function seriesDefaults(t: ChartTokens): Record<string, Dict> {
  return {
    line: { lineStyle: { width: 2, cap: "round", join: "round" }, symbolSize: 2 * t.unit, symbol: "circle" },
    bar: { barMaxWidth: 6 * t.unit },
    scatter: { symbolSize: 2 * t.unit },
  };
}

function isDict(value: unknown): value is Dict {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function merge(base: unknown, own: unknown): unknown {
  if (own === undefined) return base;
  if (isDict(base) && isDict(own)) {
    const out: Dict = { ...base };
    for (const [key, value] of Object.entries(own)) out[key] = merge(base[key], value);
    return out;
  }
  return own;
}

const ALWAYS = new Set(["color", "backgroundColor", "textStyle", "tooltip"]);

/** Replace every `token:<name>` string in an option with the token's value (functions are left alone). */
function resolveTokens(value: unknown, resolve: (name: string) => string): unknown {
  if (typeof value === "string") return value.startsWith("token:") ? resolve(value.slice("token:".length)) : value;
  if (Array.isArray(value)) return value.map((item) => resolveTokens(item, resolve));
  if (isDict(value)) {
    const out: Dict = {};
    for (const [key, item] of Object.entries(value)) out[key] = resolveTokens(item, resolve);
    return out;
  }
  return value;
}

/**
 * A chart's own option over the theme. Component defaults merge into the components the chart
 * declares (one by one when it declares several) and never add one it does not use. Every series
 * and calendar gets a stable id, so an update is matched to what is on screen and animates from
 * it instead of being redrawn as new.
 */
export function themed(option: ChartOption, t: ChartTokens): ChartOption {
  const own = option as Dict;
  const out: Dict = { ...own };
  for (const [key, style] of Object.entries(chartTheme(t))) {
    const value = own[key];
    if (value === undefined) {
      if (ALWAYS.has(key) || (key === "grid" && own.xAxis !== undefined)) out[key] = style;
      continue;
    }
    out[key] =
      Array.isArray(value) && key !== "color" ? value.map((item) => merge(style, item)) : merge(style, value);
  }
  const defaults = seriesDefaults(t);
  const series = own.series === undefined ? [] : Array.isArray(own.series) ? own.series : [own.series];
  out.series = series.map((item, index) => {
    const entry = item as Dict;
    const type = typeof entry.type === "string" ? entry.type : "";
    const merged = merge(defaults[type] ?? {}, entry) as Dict;
    return { ...merged, id: entry.id ?? entry.name ?? `series-${index}` };
  });
  if (Array.isArray(out.calendar)) {
    out.calendar = (out.calendar as Dict[]).map((item, index) => ({ ...item, id: item.id ?? `calendar-${index}` }));
  }
  return resolveTokens(out, t.resolve) as ChartOption;
}

/** HTML-escape text from the API before it enters a tooltip (ECharts tooltips are HTML). */
export function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/** One tooltip row: a short stroke of the series colour, the value strong, the label after it. */
export function tooltipRow(color: string, value: string, label: string): string {
  const key = `<span class="chart-tooltip-key" style="background:${escapeHtml(color)}"></span>`;
  return `<div class="chart-tooltip-row">${key}<strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`;
}

/** A tooltip heading line (the x position or the item's name). */
export function tooltipTitle(text: string): string {
  return `<div class="chart-tooltip-title">${escapeHtml(text)}</div>`;
}
