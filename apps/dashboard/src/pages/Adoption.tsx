import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { api } from "../api";
import { Chart, type ChartOption } from "../components/Chart";
import { Card, Failure, Kpi, Loading, PageHeader } from "../components/ui";
import { formatCount, formatShare, humanize } from "../format";

export function AdoptionPage() {
  const adoption = useQuery({ queryKey: ["adoption"], queryFn: api.adoption });

  const charts = useMemo(() => {
    const out: { target: string; option: ChartOption }[] = [];
    for (const [target, item] of Object.entries(adoption.data ?? {})) {
      out.push({
        target,
        option: {
          tooltip: { trigger: "axis" },
          legend: { top: 0 },
          xAxis: { type: "category", data: item.deciles.map((d) => `D${d.decile}`) },
          yAxis: [
            { type: "value", name: "rate", axisLabel: { formatter: (v: number) => formatShare(v, 0) } },
            { type: "value", name: "gain", min: 0, max: 1, axisLabel: { formatter: (v: number) => formatShare(v, 0) } },
          ],
          series: [
            { type: "bar", name: "rate in the decile", data: item.deciles.map((d) => d.rate), barMaxWidth: 32 },
            { type: "line", name: "cumulative gain", yAxisIndex: 1, data: item.deciles.map((d) => d.cumulative_gain) },
          ],
        },
      });
    }
    return out;
  }, [adoption.data]);

  return (
    <>
      <PageHeader
        title="Online adoption"
        lead="Who has online banking and who uses it actively, on the aggregate sample (a separate set of clients). Deciles of the model's score from out-of-fold predictions."
      />
      {adoption.isPending ? <Loading what="the adoption models" /> : null}
      {adoption.error ? <Failure error={adoption.error} /> : null}
      <div className="flex flex-col gap-4">
        {charts.map(({ target, option }) => {
          const item = adoption.data?.[target];
          return (
            <Card key={target} title={humanize(target.replace("target_", ""))}>
              <div className="mb-3 grid grid-cols-2 gap-3">
                <Kpi label="Clients" value={formatCount(item?.n_clients ?? 0)} />
                <Kpi label="Rate" value={formatShare(item?.rate)} />
              </div>
              <Chart option={option} label={`Decile chart for ${humanize(target)}`} height={280} />
            </Card>
          );
        })}
      </div>
    </>
  );
}
