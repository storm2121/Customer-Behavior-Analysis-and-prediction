import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { api, type Adoption } from "../api";
import { tooltipRow, tooltipTitle } from "../charts/theme";
import { Chart, type ChartOption } from "../components/Chart";
import { Bone, EmptyState, Failure, Loading, PageHeader, Panel, ScrollArea } from "../components/ui";
import { formatCount, formatShare, label } from "../format";

const TARGETS: Record<string, string> = {
  target_has_online: "Has online banking",
  target_active_online: "Uses online banking actively",
};

const SERIES = ["token:chart-1", "token:chart-2", "token:chart-3", "token:chart-4", "token:chart-5"];

function targetName(key: string): string {
  return TARGETS[key] ?? label(key.replace(/^target_/, ""));
}

// Cumulative gain: the share of all adopters found when contacting the top deciles first. One
// axis, one line per model, and the diagonal of contacting clients at random.
function gainOption(adoption: Adoption): ChartOption {
  const targets = Object.keys(adoption);
  const reach = (decile: number, count: number) => decile / count;
  return {
    tooltip: {
      trigger: "axis",
      formatter: (params: unknown) => {
        const items = (Array.isArray(params) ? params : [params]) as { seriesId: string; seriesName: string; value: [number, number]; color: string }[];
        const shown = items.filter((item) => item.seriesId !== "random");
        const first = shown[0];
        if (!first) return "";
        return (
          tooltipTitle(`Top ${formatShare(first.value[0], 0)} of clients contacted`) +
          shown.map((item) => tooltipRow(item.color, `${formatShare(item.value[1])} of adopters`, item.seriesName)).join("")
        );
      },
    },
    legend: { top: 0, left: 0, data: targets.map(targetName) },
    xAxis: { type: "value", min: 0, max: 1, name: "share of clients contacted, highest scores first", nameLocation: "middle", nameGap: 28, axisLabel: { formatter: (v: number) => formatShare(v, 0) } },
    yAxis: { type: "value", min: 0, max: 1, axisLabel: { formatter: (v: number) => formatShare(v, 0) } },
    series: [
      {
        type: "line",
        id: "random",
        name: "Contacting at random",
        data: [
          [0, 0],
          [1, 1],
        ],
        symbol: "none",
        silent: true,
        lineStyle: { color: "token:chart-axis", width: 1 },
        tooltip: { show: false },
      },
      ...targets.map((target, index) => {
        const deciles = adoption[target]?.deciles ?? [];
        const colour = SERIES[index % SERIES.length] ?? "token:accent";
        return {
          type: "line" as const,
          id: target,
          name: targetName(target),
          data: [[0, 0], ...deciles.map((decile) => [reach(decile.decile, deciles.length), decile.cumulative_gain])],
          itemStyle: { color: colour },
          lineStyle: { color: colour },
        };
      }),
    ],
  } as ChartOption;
}

export function AdoptionPage() {
  const adoption = useQuery({ queryKey: ["adoption"], queryFn: api.adoption });
  const data = adoption.data;
  const gain = useMemo(() => (data ? gainOption(data) : null), [data]);
  const targets = data ? Object.keys(data) : [];

  return (
    <>
      <PageHeader
        title="Adoption"
        lead="Who has online banking and who uses it actively, on the aggregate sample (a separate set of clients). Each model's score is cut into deciles from out-of-fold predictions; the first decile holds the highest scores."
      />
      {adoption.error ? <Failure error={adoption.error} onRetry={() => void adoption.refetch()} /> : null}
      {adoption.isPending ? (
        <Loading what="the adoption models">
          <div className="grid gap-5 xl:grid-cols-2">
            <Bone className="h-96 w-full" />
            <Bone className="h-96 w-full" />
          </div>
          <Bone className="h-72 w-full" />
        </Loading>
      ) : null}
      {data && targets.length === 0 ? (
        <EmptyState title="No adoption model is trained yet." command="just train">
          The decile tables appear once the adoption models are trained on the aggregate sample.
        </EmptyState>
      ) : null}
      {data && gain && targets.length ? (
        <div className="flex flex-col gap-5">
          <div className="grid grid-cols-1 items-start gap-5 xl:grid-cols-2">
            {targets.map((target) => {
              const item = data[target];
              if (!item) return null;
              return (
                <Panel
                  key={target}
                  title={targetName(target)}
                  subtitle={`${formatShare(item.rate)} of ${formatCount(item.n_clients)} clients. Lift is the decile's rate over that average.`}
                >
                  <ScrollArea label={`Deciles for ${targetName(target)}`}>
                    <table className="data-table min-w-[28rem]">
                      <thead>
                        <tr>
                          <th scope="col">Decile</th>
                          <th scope="col" className="num">
                            Clients
                          </th>
                          <th scope="col" className="num">
                            Rate
                          </th>
                          <th scope="col" className="num">
                            Lift
                          </th>
                          <th scope="col" className="num">
                            Cumulative gain
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {item.deciles.map((decile) => (
                          <tr key={decile.decile}>
                            <td>{formatCount(decile.decile)}</td>
                            <td className="num">{formatCount(decile.n)}</td>
                            <td className="num">{formatShare(decile.rate)}</td>
                            <td className="num">{`${decile.lift.toFixed(2)}×`}</td>
                            <td className="num">{formatShare(decile.cumulative_gain)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </ScrollArea>
                </Panel>
              );
            })}
          </div>
          <Panel title="Cumulative gain" subtitle="The share of all adopters found when the clients with the highest scores are contacted first, against contacting them at random (the diagonal).">
            <Chart option={gain} label="Line chart of cumulative gain against the share of clients contacted, for each adoption model" />
          </Panel>
        </div>
      ) : null}
    </>
  );
}
