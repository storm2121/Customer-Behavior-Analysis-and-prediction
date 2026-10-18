import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { api } from "../api";
import { Chart, type ChartOption } from "../components/Chart";
import { Card, Failure, Kpi, Loading, PageHeader, ScrollArea } from "../components/ui";
import { formatCount, formatNumber, formatShare } from "../format";

export function SegmentsPage() {
  const segments = useQuery({ queryKey: ["segments"], queryFn: api.segments });
  const points = useQuery({ queryKey: ["segment-points"], queryFn: api.segmentPoints });

  const scatter = useMemo<ChartOption>(() => {
    const groups = new Map<number, [number, number][]>();
    for (const point of points.data?.points ?? []) {
      const list = groups.get(point.segment) ?? [];
      list.push([point.x, point.y]);
      groups.set(point.segment, list);
    }
    const names = new Map((segments.data?.segments ?? []).map((s) => [s.segment, s.name]));
    return {
      tooltip: { trigger: "item" },
      legend: { type: "scroll", bottom: 0 },
      xAxis: { type: "value", show: false, scale: true },
      yAxis: { type: "value", show: false, scale: true },
      series: [...groups.entries()]
        .sort(([a], [b]) => a - b)
        .map(([segment, data]) => ({
          type: "scatter" as const,
          name: names.get(segment) ?? `segment ${segment}`,
          data,
          symbolSize: 5,
        })),
    };
  }, [points.data, segments.data]);

  return (
    <>
      <PageHeader
        title="Segments"
        lead="Behavioural segments of active clients from Gaussian mixtures chosen by BIC, with their stability over bootstrap refits; the map is a two-dimensional embedding for display only."
      />
      {segments.isPending ? <Loading what="the segments" /> : null}
      {segments.error ? <Failure error={segments.error} /> : null}
      {segments.data ? (
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Kpi label="Segments" value={String(segments.data.k)} hint="chosen by BIC" />
            <Kpi
              label="Stability (adjusted Rand index)"
              value={formatNumber(segments.data.stability.mean_ari, 2)}
              hint={`95% interval ${formatNumber(segments.data.stability.ci_low, 2)} to ${formatNumber(segments.data.stability.ci_high, 2)}`}
            />
          </div>
          <Card title="Map of clients" subtitle="Each dot is a synthetic client, coloured by segment.">
            <Chart option={scatter} label="Scatter plot of clients in two dimensions, coloured by segment" height={360} />
          </Card>
          <Card title="Profiles">
            <ScrollArea label="Segment profiles">
              <table className="w-full min-w-[28rem] text-left text-sm">
                <thead className="text-xs uppercase tracking-wide text-ink-muted">
                  <tr>
                    <th className="py-2 pr-3">Segment</th>
                    <th className="py-2 pr-3">Name</th>
                    <th className="py-2 pr-3 text-right">Clients</th>
                    <th className="py-2 text-right">Share</th>
                  </tr>
                </thead>
                <tbody>
                  {segments.data.segments.map((segment) => (
                    <tr key={segment.segment} className="border-t border-border">
                      <td className="py-2 pr-3">#{segment.segment}</td>
                      <td className="py-2 pr-3">{segment.name}</td>
                      <td className="py-2 pr-3 text-right tabular-nums">{formatCount(segment.n_clients)}</td>
                      <td className="py-2 text-right tabular-nums">{formatShare(segment.share)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ScrollArea>
          </Card>
        </div>
      ) : null}
    </>
  );
}
