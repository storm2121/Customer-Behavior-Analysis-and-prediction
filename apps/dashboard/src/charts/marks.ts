// Marks the stock series do not draw: an interval as a thin whisker with short caps, for bars and
// dot plots whose numbers carry their uncertainty. The colour is a token name, resolved with the
// rest of the option on the chart's own element.

type Dict = Record<string, unknown>;

interface RenderApi {
  value: (dimension: number) => number;
  coord: (point: [number, number]) => [number, number];
  size: (point: [number, number]) => [number, number];
  visual: (name: string) => string;
}

/**
 * A custom series drawing one whisker per category: from `low` to `high` along the value axis, at
 * the category's position. `orient` says which axis holds the values ("horizontal": values on x).
 */
export function intervalSeries({
  id,
  intervals,
  orient = "horizontal",
  colour = "token:ink-secondary",
}: {
  id: string;
  intervals: ([number, number] | null)[];
  orient?: "horizontal" | "vertical";
  colour?: string;
}): Dict {
  const horizontal = orient === "horizontal";
  return {
    type: "custom",
    id,
    silent: true,
    z: 4,
    itemStyle: { color: colour },
    tooltip: { show: false },
    encode: horizontal ? { x: [1, 2], y: 0 } : { x: 0, y: [1, 2] },
    data: intervals.flatMap((interval, index) => (interval ? [[index, interval[0], interval[1]]] : [])),
    renderItem: (_params: unknown, api: RenderApi) => {
      const category = api.value(0);
      const low = api.value(1);
      const high = api.value(2);
      const start = horizontal ? api.coord([low, category]) : api.coord([category, low]);
      const end = horizontal ? api.coord([high, category]) : api.coord([category, high]);
      const band = horizontal ? api.size([0, 1])[1] : api.size([1, 0])[0];
      const cap = Math.max(2, band * 0.18);
      const stroke = api.visual("color");
      const style = { stroke, lineWidth: 1.5, lineCap: "round" };
      const line = (x1: number, y1: number, x2: number, y2: number) => ({ type: "line", shape: { x1, y1, x2, y2 }, style });
      return {
        type: "group",
        children: horizontal
          ? [
              line(start[0], start[1], end[0], end[1]),
              line(start[0], start[1] - cap, start[0], start[1] + cap),
              line(end[0], end[1] - cap, end[0], end[1] + cap),
            ]
          : [
              line(start[0], start[1], end[0], end[1]),
              line(start[0] - cap, start[1], start[0] + cap, start[1]),
              line(end[0] - cap, end[1], end[0] + cap, end[1]),
            ],
      };
    },
  };
}

/**
 * A reference line across the whole plot at `at` on the value axis (no lift at one, for instance),
 * drawn as a quiet hairline under the data.
 */
export function referenceLine({
  id,
  at,
  categories,
  orient = "horizontal",
  colour = "token:chart-axis",
}: {
  id: string;
  at: number;
  categories: number;
  orient?: "horizontal" | "vertical";
  colour?: string;
}): Dict {
  const horizontal = orient === "horizontal";
  return {
    type: "custom",
    id,
    silent: true,
    z: 1,
    itemStyle: { color: colour },
    tooltip: { show: false },
    encode: horizontal ? { x: 1, y: 0 } : { x: 0, y: 1 },
    data: categories > 0 ? [[0, at, categories - 1]] : [],
    renderItem: (_params: unknown, api: RenderApi) => {
      const last = api.value(2);
      const value = api.value(1);
      const start = horizontal ? api.coord([value, 0]) : api.coord([0, value]);
      const end = horizontal ? api.coord([value, last]) : api.coord([last, value]);
      const band = horizontal ? api.size([0, 1])[1] : api.size([1, 0])[0];
      const reach = band / 2;
      const shape = horizontal
        ? { x1: start[0], y1: Math.min(start[1], end[1]) - reach, x2: end[0], y2: Math.max(start[1], end[1]) + reach }
        : { x1: Math.min(start[0], end[0]) - reach, y1: start[1], x2: Math.max(start[0], end[0]) + reach, y2: end[1] };
      return { type: "line", shape, style: { stroke: api.visual("color"), lineWidth: 1 } };
    },
  };
}
