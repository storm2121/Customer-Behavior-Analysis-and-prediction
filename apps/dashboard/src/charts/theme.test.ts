import { chartTheme, escapeHtml, readChartTokens, themed, type ChartOption } from "./theme";

// A fake token reader: every token resolves to its own name, so the test can see what came from where.
const tokens = readChartTokens(document.documentElement, (name) => {
  if (name === "type-label-size") return "12px";
  if (name === "space-unit") return "4px";
  if (name === "duration-chart") return "300ms";
  if (name === "duration-chart-enter") return "0.5s";
  return `#${name}`;
});

describe("the chart theme", () => {
  it("is generated from the tokens: categorical slots in order, text in text tokens", () => {
    const theme = chartTheme(tokens) as Record<string, unknown>;
    expect(theme.color).toEqual(["#chart-1", "#chart-2", "#chart-3", "#chart-4", "#chart-5"]);
    expect(theme.textStyle).toMatchObject({ color: "#chart-label", fontSize: 12 });
    expect(tokens.update).toBe(300);
    expect(tokens.enter).toBe(500);
  });

  it("merges defaults only into the components a chart declares, and gives every series an id", () => {
    const option = themed({ series: [{ type: "bar", data: [1] }, { type: "line", name: "policy", data: [2] }] } as ChartOption, tokens) as Record<string, unknown>;
    expect(option.xAxis).toBeUndefined();
    expect(option.calendar).toBeUndefined();
    const series = option.series as Record<string, unknown>[];
    expect(series.map((item) => item.id)).toEqual(["series-0", "policy"]);
    expect(series[1]).toMatchObject({ lineStyle: { width: 2 } });
  });

  it("resolves colours named by token, anywhere in the option, and leaves functions alone", () => {
    const formatter = () => "x";
    const option = themed(
      { xAxis: { type: "value", axisLabel: { formatter } }, series: [{ type: "bar", data: [{ value: 1, itemStyle: { color: "token:state-mixed" } }] }] } as ChartOption,
      tokens,
    ) as Record<string, unknown>;
    const series = option.series as { data: { itemStyle: { color: string } }[] }[];
    expect(series[0]?.data[0]?.itemStyle.color).toBe("#state-mixed");
    expect((option.xAxis as { axisLabel: { formatter: unknown } }).axisLabel.formatter).toBe(formatter);
    expect((option.xAxis as { axisLine: { lineStyle: { color: string } } }).axisLine.lineStyle.color).toBe("#chart-axis");
  });

  it("escapes text from the API before it enters a tooltip", () => {
    expect(escapeHtml(`<b>"segment" & 'name'</b>`)).toBe("&lt;b&gt;&quot;segment&quot; &amp; &#39;name&#39;&lt;/b&gt;");
  });
});
