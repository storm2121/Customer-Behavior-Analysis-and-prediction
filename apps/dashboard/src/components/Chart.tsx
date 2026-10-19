import { BarChart, CustomChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
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

import { readChartTokens, themed, type ChartOption } from "../charts/theme";
import { prefersReducedMotion } from "../motion";
import { useThemeKey } from "../theme";

// Only what the pages draw, so the bundle carries a fraction of ECharts.
echarts.use([
  BarChart,
  CustomChart,
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

export type { ChartOption };

/** Chart heights, named so a page never types a pixel value. */
const HEIGHTS = {
  compact: "h-56",
  standard: "h-72",
  tall: "h-96",
} as const;

export type ChartHeight = keyof typeof HEIGHTS;

export type ChartEvents = Partial<Record<"mouseover" | "mouseout" | "click" | "globalout", (params: unknown) => void>>;

// Components replaced (matched by id) rather than merged, so a series or calendar that leaves the
// option leaves the chart, and one that stays is updated in place and animates from the screen.
const REPLACED = ["series", "calendar", "visualMap"];

/**
 * An ECharts canvas that keeps one instance for its whole life. Data changes go through ECharts'
 * own transitions (series matched by id); the entrance animation plays only on the first paint;
 * `transition="immediate"` draws without easing, for values that follow a pointer; reduced motion
 * turns animation off. The chart follows its container's size and the theme in effect on it.
 */
export function Chart({
  option,
  label,
  height = "standard",
  pixels,
  transition = "smooth",
  events,
}: {
  option: ChartOption;
  label: string;
  height?: ChartHeight;
  /** A height computed from the chart's own geometry (a calendar's cell size), in CSS pixels. */
  pixels?: number;
  transition?: "smooth" | "immediate";
  events?: ChartEvents;
}) {
  const element = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.EChartsType | null>(null);
  const painted = useRef(false);
  const themeKey = useThemeKey();

  useEffect(() => {
    if (!element.current) return;
    const host = element.current;
    const instance = echarts.init(host, undefined, { renderer: "canvas" });
    chart.current = instance;
    // The screenshot script waits for every chart to finish drawing, animations included.
    const finished = () => {
      host.dataset.settled = "true";
    };
    instance.on("finished", finished);
    const observer = new ResizeObserver(() => instance.resize({ animation: { duration: 0 } }));
    observer.observe(element.current);
    return () => {
      observer.disconnect();
      instance.off("finished", finished);
      instance.dispose();
      chart.current = null;
      painted.current = false;
    };
  }, []);

  useEffect(() => {
    const instance = chart.current;
    const host = element.current;
    if (!instance || !host) return;
    const tokens = readChartTokens(host);
    const first = !painted.current;
    const motion = prefersReducedMotion()
      ? { animation: false }
      : {
          animation: true,
          animationDuration: first ? tokens.enter : 0,
          animationEasing: "cubicOut" as const,
          animationDurationUpdate: transition === "immediate" ? 0 : tokens.update,
          animationEasingUpdate: "cubicInOut" as const,
        };
    host.dataset.settled = "false";
    instance.setOption({ ...themed(option, tokens), ...motion }, { replaceMerge: REPLACED, lazyUpdate: transition === "immediate" });
    painted.current = true;
  }, [option, themeKey, transition]);

  useEffect(() => {
    const instance = chart.current;
    if (!instance || !events) return;
    const entries = Object.entries(events) as [keyof ChartEvents, (params: unknown) => void][];
    for (const [name, handler] of entries) {
      if (name === "globalout") instance.getZr().on("globalout", handler);
      else instance.on(name, handler);
    }
    return () => {
      for (const [name, handler] of entries) {
        if (name === "globalout") instance.getZr().off("globalout", handler);
        else instance.off(name, handler);
      }
    };
  }, [events]);

  return (
    <div
      ref={element}
      role="img"
      aria-label={label}
      data-chart=""
      className={`w-full ${pixels === undefined ? HEIGHTS[height] : ""}`}
      style={pixels === undefined ? undefined : { height: pixels }}
    />
  );
}
