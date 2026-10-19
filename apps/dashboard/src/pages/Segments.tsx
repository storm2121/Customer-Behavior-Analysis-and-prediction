import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { api, type Segment, type SegmentPoint } from "../api";
import { tooltipRow, tooltipTitle } from "../charts/theme";
import { Chart, type ChartEvents, type ChartOption } from "../components/Chart";
import { Bone, EmptyState, Failure, Loading, PageHeader, Panel, ScrollArea } from "../components/ui";
import { formatCount, formatFixed, formatShare } from "../format";

const EMBEDDING: Record<string, string> = {
  umap: "a two-dimensional UMAP embedding",
  pca: "the first two principal components",
};

const SEGMENT_ID = /^segment-(\d+)$/;

// The map shows every segment in the neutral ramp and the one under discussion in the accent: a
// segmentation can have more segments than five hues can tell apart in a scatter, and the table
// beside the map names them all.
function mapOption(points: SegmentPoint[], segments: Segment[], active: number | null): ChartOption {
  const names = new Map(segments.map((segment) => [segment.segment, segment.name]));
  const groups = new Map<number, [number, number, string][]>();
  for (const point of points) {
    const list = groups.get(point.segment) ?? [];
    list.push([point.x, point.y, point.client_id]);
    groups.set(point.segment, list);
  }
  return {
    tooltip: {
      trigger: "item",
      formatter: (item: unknown) => {
        const { seriesId, value, color } = item as { seriesId: string; value: [number, number, string]; color: string };
        const segment = Number(SEGMENT_ID.exec(seriesId)?.[1]);
        return tooltipTitle(`Segment ${segment}: ${names.get(segment) ?? ""}`) + tooltipRow(color, value[2], "synthetic client");
      },
    },
    grid: { left: 0, right: 0, top: 0, bottom: 0, containLabel: false },
    xAxis: { type: "value", show: false, scale: true },
    yAxis: { type: "value", show: false, scale: true },
    series: [...groups.entries()]
      .sort(([a], [b]) => a - b)
      .map(([segment, data]) => {
        const on = segment === active;
        return {
          type: "scatter" as const,
          id: `segment-${segment}`,
          name: names.get(segment) ?? `segment ${segment}`,
          data,
          symbolSize: on ? 7 : 5,
          z: on ? 3 : 2,
          itemStyle: { color: on ? "token:accent" : "token:chart-neutral", opacity: on ? 0.9 : 0.35 },
          emphasis: { scale: 1.6, itemStyle: { opacity: 1 } },
        };
      }),
  } as ChartOption;
}

export function SegmentsPage() {
  const segments = useQuery({ queryKey: ["segments"], queryFn: api.segments });
  const points = useQuery({ queryKey: ["segment-points"], queryFn: api.segmentPoints });
  const [chosen, setChosen] = useState<number | null>(null);

  const rows = useMemo(() => [...(segments.data?.segments ?? [])].sort((a, b) => b.n_clients - a.n_clients), [segments.data]);
  const active = chosen ?? rows[0]?.segment ?? null;
  const option = useMemo(
    () => (points.data && segments.data ? mapOption(points.data.points, segments.data.segments, active) : null),
    [points.data, segments.data, active],
  );
  const events = useMemo<ChartEvents>(
    () => ({
      mouseover: (params) => {
        const id = (params as { seriesId?: string }).seriesId ?? "";
        const match = SEGMENT_ID.exec(id);
        if (match) setChosen(Number(match[1]));
      },
    }),
    [],
  );

  const data = segments.data;
  const embedding = data?.embedding ? EMBEDDING[data.embedding] : null;
  const activeRow = rows.find((row) => row.segment === active);

  return (
    <>
      <PageHeader
        title="Segments"
        lead={`Behavioural segments of active clients from Gaussian mixtures chosen by BIC, with their stability over bootstrap refits. The map is ${embedding ?? "a two-dimensional embedding"} of the same features, for display only.`}
      />
      {segments.error ? <Failure error={segments.error} onRetry={() => void segments.refetch()} /> : null}
      {points.error ? <Failure error={points.error} onRetry={() => void points.refetch()} /> : null}
      {segments.isPending ? (
        <Loading what="the segments">
          <div className="grid gap-5 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
            <Bone className="h-96 w-full" />
            <Bone className="h-96 w-full" />
          </div>
        </Loading>
      ) : null}
      {data ? (
        <div className="grid items-start gap-5 xl:grid-cols-2">
          <Panel
            title="Map of clients"
            subtitle={activeRow ? `Each dot is a synthetic client; segment ${activeRow.segment} is in colour. Point at a dot or a row to show another.` : "Each dot is a synthetic client."}
          >
            {option ? (
              points.data && points.data.points.length ? (
                <Chart option={option} height="tall" events={events} label={`Scatter plot of synthetic clients on ${embedding ?? "a two-dimensional embedding"}, segment ${active ?? ""} highlighted`} />
              ) : (
                <EmptyState title="No map for these segments." command="just train">
                  The map appears once the segments are trained with their embedding.
                </EmptyState>
              )
            ) : (
              <Bone className="h-96 w-full" />
            )}
          </Panel>
          <Panel
            title="Profiles"
            subtitle={
              <>
                {formatCount(data.k)} segments chosen by BIC{data.n_clients ? ` over ${formatCount(data.n_clients)} active clients` : ""}. Stability
                (adjusted Rand index){" "}
                <span className="font-label text-ink">
                  {formatFixed(data.stability.mean_ari, [data.stability.ci_low, data.stability.ci_high], 2)}
                </span>
                , the {data.stability.interval}.
              </>
            }
          >
            <ScrollArea label="Segment profiles">
              <table className="data-table compact min-w-[26rem]">
                <thead>
                  <tr>
                    <th scope="col">Segment</th>
                    <th scope="col" className="num">
                      Clients
                    </th>
                    <th scope="col" className="num">
                      Share
                    </th>
                    <th scope="col" className="num">
                      Card
                    </th>
                    <th scope="col" className="num">
                      Online
                    </th>
                    <th scope="col" className="num">
                      Branch
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((segment) => {
                    const on = segment.segment === active;
                    return (
                      <tr
                        key={segment.segment}
                        tabIndex={0}
                        aria-current={on ? "true" : undefined}
                        onMouseEnter={() => setChosen(segment.segment)}
                        onFocus={() => setChosen(segment.segment)}
                        className={`cursor-default outline-none focus-visible:bg-accent-tint ${on ? "bg-accent-tint" : "hover:bg-fill-subtle"}`}
                      >
                        <td>
                          <span className="flex items-center gap-2">
                            <span aria-hidden className={`size-2.5 shrink-0 rounded-full ${on ? "bg-accent" : "bg-chart-neutral"}`} />
                            <span className="flex min-w-0 flex-col py-1.5">
                              <span className="font-label text-ink">Segment {segment.segment}</span>
                              <span className="text-label text-ink-secondary">{segment.name}</span>
                            </span>
                          </span>
                        </td>
                        <td className="num">{formatCount(segment.n_clients)}</td>
                        <td className="num">{formatShare(segment.share)}</td>
                        <td className="num">{formatShare(segment.medians.card_share)}</td>
                        <td className="num">{formatShare(segment.medians.online_share)}</td>
                        <td className="num">{formatShare(segment.medians.branch_share)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </ScrollArea>
            <p className="mt-3 text-label text-ink-secondary">Card, online and branch: the median share of each client's events in that channel.</p>
          </Panel>
        </div>
      ) : null}
    </>
  );
}
