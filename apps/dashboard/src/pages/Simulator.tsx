import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import { api, type Bands, type SimulationRequest } from "../api";
import { Chart, type ChartOption } from "../components/Chart";
import { Slider } from "../components/Controls";
import { Card, Failure, Loading, PageHeader, Tag } from "../components/ui";
import { formatCount, formatShare } from "../format";
import { token, useThemeKey } from "../theme";

const AGES = [25, 35, 45, 55, 65];

function useDebounced<T>(value: T, delay = 250): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delay);
    return () => window.clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}

/**
 * A fan chart: the posterior median of `quantity` for the policy and the baseline, each with its
 * 90% band (drawn as a stacked area from the 5th to the 95th percentile).
 */
function fan(months: number[], policy: Bands, baseline: Bands, quantity: string, share: boolean): ChartOption {
  const format = (value: number | undefined) =>
    value === undefined ? "not defined" : share ? formatShare(value, 1) : formatCount(value);
  const band = (bands: Bands, name: string, color: string) => {
    const q = bands[quantity];
    const low = q?.p05 ?? [];
    return [
      { type: "line" as const, name: `${name} low`, data: low, stack: name, symbol: "none", lineStyle: { opacity: 0 }, silent: true },
      {
        type: "line" as const,
        name: `${name} band`,
        data: (q?.p95 ?? []).map((high, index) => high - (low[index] ?? high)),
        stack: name,
        symbol: "none",
        lineStyle: { opacity: 0 },
        areaStyle: { color, opacity: 0.16 },
        silent: true,
      },
      {
        type: "line" as const,
        name,
        data: q?.p50 ?? [],
        symbol: "none",
        itemStyle: { color },
        lineStyle: { color, width: 2.5 },
      },
    ];
  };
  const row = (bands: Bands, name: string, index: number) => {
    const q = bands[quantity];
    return `${name}: ${format(q?.p50[index])} (90% band ${format(q?.p05[index])} to ${format(q?.p95[index])})`;
  };
  return {
    tooltip: {
      trigger: "axis",
      formatter: (params: unknown) => {
        const first = (Array.isArray(params) ? params[0] : params) as { dataIndex?: number } | undefined;
        const index = first?.dataIndex ?? 0;
        return [`month ${months[index] ?? index}`, row(policy, "policy", index), row(baseline, "baseline", index)].join("<br/>");
      },
    },
    legend: { top: 0, data: ["baseline", "policy"] },
    xAxis: { type: "category", data: months.map((month) => `${month}`), name: "months", boundaryGap: false },
    yAxis: { type: "value", scale: true, axisLabel: { formatter: (value: number) => format(value) } },
    series: [
      ...band(baseline, "baseline", token("chart-6") || "#586372"),
      ...band(policy, "policy", token("chart-1") || "#2f6f9f"),
    ],
  };
}

export function SimulatorPage() {
  const [share, setShare] = useState(0.3);
  const [maxAge, setMaxAge] = useState(45);
  const [target, setTarget] = useState<SimulationRequest["target"]>("mixed");
  const request = useDebounced<SimulationRequest>({ share, max_age: maxAge, target, horizon: 12, samples: 600 });
  const simulation = useQuery({
    queryKey: ["simulate", request],
    queryFn: () => api.simulate(request),
    placeholderData: (previous) => previous,
  });

  const theme = useThemeKey();
  const charts = useMemo(() => {
    if (!simulation.data) return null;
    const { months, policy_bands: policy, baseline_bands: baseline } = simulation.data;
    return {
      digital: fan(months, policy, baseline, "digital_share", true),
      branch: fan(months, policy, baseline, "branch_load", false),
    };
  }, [simulation.data, theme]);

  return (
    <>
      <PageHeader
        title="What-if simulator"
        lead="A Markov chain of monthly channel states with a Dirichlet posterior. Move a share of the branch-only clients under an age limit to another state this month and watch the next year, with 90% bands from the posterior."
      />
      <Card title="Policy">
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
          <Slider
            label="Share of eligible clients who move"
            value={share}
            min={0}
            max={1}
            step={0.05}
            display={formatShare(share, 0)}
            onChange={setShare}
          />
          <Slider
            label="Clients younger than"
            value={maxAge}
            min={AGES[0] ?? 25}
            max={AGES[AGES.length - 1] ?? 65}
            step={10}
            display={`${maxAge} years`}
            onChange={setMaxAge}
          />
          <label className="flex flex-col gap-2 text-sm">
            <span className="font-medium">They move to</span>
            <select
              className="rounded-md border border-border bg-surface px-3 py-2 text-ink"
              value={target}
              onChange={(event) => setTarget(event.target.value as SimulationRequest["target"])}
            >
              <option value="mixed">mixed use (branch and digital)</option>
              <option value="digital_only">digital only</option>
            </select>
          </label>
        </div>
        {simulation.data ? (
          <p className="mt-4 flex flex-wrap gap-2">
            <Tag>Eligible age bands: {simulation.data.eligible_bands.join(", ") || "none"}</Tag>
            <Tag>A what-if under the fitted chain, not a forecast</Tag>
          </p>
        ) : null}
      </Card>
      {simulation.isPending ? <Loading what="the simulation" /> : null}
      {simulation.error ? <Failure error={simulation.error} /> : null}
      {charts ? (
        <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Card title="Digital share" subtitle="Clients using card or online banking in the month.">
            <Chart option={charts.digital} label="Digital share over twelve months, policy against baseline, with bands" height={300} />
          </Card>
          <Card title="Branch operations" subtitle="Expected branch operations in the month.">
            <Chart option={charts.branch} label="Branch operations over twelve months, policy against baseline, with bands" height={300} />
          </Card>
        </div>
      ) : null}
    </>
  );
}
