import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { api } from "../api";
import { Chart, type ChartOption } from "../components/Chart";
import { Card, Failure, Kpi, Loading, PageHeader } from "../components/ui";
import { formatCount, formatShare, humanize } from "../format";
import { token, useThemeKey } from "../theme";

export function OverviewPage() {
  const overview = useQuery({ queryKey: ["overview"], queryFn: api.overview });
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const theme = useThemeKey();

  const mix = useMemo<ChartOption>(() => {
    const rows = overview.data?.channel_mix ?? [];
    const total = rows.reduce((sum, row) => sum + row.clients, 0) || 1;
    return {
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: rows.map((row) => humanize(row.channel_profile)) },
      yAxis: { type: "value", axisLabel: { formatter: (value: number) => formatShare(value, 0) } },
      series: [{ type: "bar", data: rows.map((row) => row.clients / total), barMaxWidth: 56 }],
    };
  }, [overview.data]);

  const heatmap = useMemo<ChartOption>(() => {
    const byDay = new Map<string, number>();
    for (const row of overview.data?.daily ?? []) {
      byDay.set(row.day.slice(0, 10), (byDay.get(row.day.slice(0, 10)) ?? 0) + row.events);
    }
    const days = [...byDay.entries()].sort(([a], [b]) => a.localeCompare(b));
    const values = days.map(([, value]) => value);
    const first = days[0]?.[0] ?? "2024-01-01";
    const last = days[days.length - 1]?.[0] ?? "2024-12-31";
    return {
      tooltip: { formatter: (item: unknown) => {
        const value = (item as { value: [string, number] }).value;
        return `${value[0]}: ${formatCount(value[1])} events`;
      } },
      visualMap: {
        min: Math.min(...values, 0),
        max: Math.max(...values, 1),
        orient: "horizontal",
        left: "center",
        bottom: 0,
        calculable: false,
        inRange: { color: [token("accent-soft") || "#dbe9f4", token("accent") || "#2f6f9f"] },
      },
      calendar: { range: [first, last], cellSize: ["auto", 14], top: 24, left: 36, right: 12, yearLabel: { show: false } },
      series: [{ type: "heatmap", coordinateSystem: "calendar", data: days }],
    };
  }, [overview.data, theme]);

  return (
    <>
      <PageHeader
        title="Overview"
        lead="The reference pipeline on synthetic clients: how many there are, which channels they use, and how activity moves through the year."
      />
      {overview.isPending ? <Loading what="the overview" /> : null}
      {overview.error ? <Failure error={overview.error} /> : null}
      {overview.data ? (
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <Kpi label="Clients" value={formatCount(overview.data.clients)} hint="in the client file" />
            <Kpi label="Events" value={formatCount(overview.data.events)} hint="card, online and branch" />
            <Kpi
              label="Data"
              value={overview.data.source_kind}
              hint={health.data ? `fingerprint ${health.data.data.fingerprint.slice(0, 12)}` : undefined}
            />
          </div>
          <Card title="Channel mix" subtitle="Share of clients by the channels they used over the year.">
            <Chart option={mix} label="Bar chart of the share of clients in each channel profile" height={280} />
          </Card>
          <Card title="Daily activity" subtitle="Events per day across all channels; darker means busier.">
            <Chart option={heatmap} label="Calendar heatmap of daily events" height={240} />
          </Card>
        </div>
      ) : null}
    </>
  );
}
