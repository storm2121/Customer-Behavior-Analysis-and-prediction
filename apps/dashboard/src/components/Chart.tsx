import type { EChartsOption } from "echarts";
import { BarChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  CalendarComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useRef } from "react";

import { token, useThemeKey } from "../theme";

// Only what the pages draw, so the bundle carries a fraction of ECharts.
echarts.use([
  BarChart,
  HeatmapChart,
  LineChart,
  ScatterChart,
  CalendarComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

export type ChartOption = EChartsOption;

type Dict = Record<string, unknown>;

const SERIES_TOKENS = ["chart-1", "chart-2", "chart-3", "chart-4", "chart-5", "chart-6"];

function isDict(value: unknown): value is Dict {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Colours and text styles from the design tokens, read when called, so they follow the theme. */
export function baseStyles(): Dict {
  const ink = token("ink") || "#17202b";
  const muted = token("ink-muted") || "#556171";
  const border = token("border") || "#d9dee6";
  const surface = token("surface") || "#ffffff";
  const axis = {
    axisLine: { lineStyle: { color: border } },
    axisTick: { lineStyle: { color: border } },
    axisLabel: { color: muted },
    nameTextStyle: { color: muted },
    splitLine: { lineStyle: { color: border } },
  };
  return {
    color: SERIES_TOKENS.map((name) => token(name) || "#2f6f9f"),
    textStyle: { color: ink, fontFamily: "inherit" },
    tooltip: { backgroundColor: surface, borderColor: border, textStyle: { color: ink } },
    legend: { textStyle: { color: muted }, pageTextStyle: { color: muted }, inactiveColor: border },
    grid: { left: 48, right: 16, top: 32, bottom: 40, containLabel: true },
    xAxis: axis,
    yAxis: axis,
    visualMap: { textStyle: { color: muted } },
    calendar: {
      dayLabel: { color: muted },
      monthLabel: { color: muted },
      yearLabel: { color: muted },
      itemStyle: { color: surface, borderColor: token("canvas") || "#f7f8fa" },
      splitLine: { lineStyle: { color: border } },
    },
  };
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

const ALWAYS = new Set(["color", "textStyle", "tooltip"]);

/**
 * A chart's own option over the token styles. Component styles (axes, legend, calendar, ...) are
 * merged into the components the chart declares, one by one when it declares several, and never
 * add a component the chart does not use.
 */
export function themed(option: ChartOption): ChartOption {
  const own = option as Dict;
  const out: Dict = { ...own };
  for (const [key, style] of Object.entries(baseStyles())) {
    const value = own[key];
    if (value === undefined) {
      if (ALWAYS.has(key) || (key === "grid" && own.xAxis !== undefined)) out[key] = style;
      continue;
    }
    out[key] =
      Array.isArray(value) && key !== "color" ? value.map((item) => merge(style, item)) : merge(style, value);
  }
  return out as ChartOption;
}

/** An ECharts canvas that follows its container's size and the page's theme. */
export function Chart({
  option,
  height = 320,
  label,
}: {
  option: ChartOption;
  height?: number;
  label: string;
}) {
  const element = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.EChartsType | null>(null);
  const theme = useThemeKey();

  useEffect(() => {
    if (!element.current) return;
    const instance = echarts.init(element.current, undefined, { renderer: "canvas" });
    chart.current = instance;
    const observer = new ResizeObserver(() => instance.resize());
    observer.observe(element.current);
    return () => {
      observer.disconnect();
      instance.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    chart.current?.setOption(themed(option), { notMerge: true });
  }, [option, theme]);

  return <div ref={element} role="img" aria-label={label} style={{ height, width: "100%" }} />;
}
