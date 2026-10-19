import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { api, type Overview } from "../api";
import { CHANNEL_STATES, stateLabel } from "../channels";
import { intervalSeries } from "../charts/marks";
import { tooltipRow, tooltipTitle } from "../charts/theme";
import { Chart, type ChartOption } from "../components/Chart";
import { Bone, Failure, Figure, Loading, PageHeader, Panel } from "../components/ui";
import { formatCount, formatInterval, formatPeriod, formatShare } from "../format";
import { useElementWidth } from "../hooks";

const FIGURES: { key: string; label: string; context: string }[] = [
  { key: "mixed", label: "Mixed use", context: "of clients used the branch and a digital channel during the year" },
  { key: "digital_only", label: "Digital only", context: "used card or online banking and never the branch" },
  { key: "branch_only", label: "Branch only", context: "used the branch and no digital channel" },
];

interface Share {
  value: number;
  ci: [number, number] | null;
}

/** Each profile's share, with the ledger's interval when the zone's ledger has one. */
function shares(overview: Overview): Record<string, Share> {
  const total = overview.channel_mix.reduce((sum, row) => sum + row.clients, 0);
  const fromLedger = overview.channel_mix_intervals?.shares ?? {};
  const out: Record<string, Share> = {};
  for (const row of overview.channel_mix) {
    const ledger = fromLedger[row.channel_profile];
    out[row.channel_profile] = ledger
      ? { value: ledger.value, ci: ledger.ci }
      : { value: total > 0 ? row.clients / total : Number.NaN, ci: null };
  }
  return out;
}

// Chart geometry in CSS pixels (a canvas wants numbers): room for the share labels at the bar
// tips, the calendar's day and month labels, its legend, and the cell size range that keeps a day
// visible on a phone and calm on a wide screen.
const MIX_RIGHT = 16;
const CALENDAR = { left: 32, right: 8, top: 24, gap: 36, minCell: 8, maxCell: 18, legend: 40, legendLength: 120, legendThickness: 10 };
const PHONE = 640;

const dayFormat = new Intl.DateTimeFormat("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });

function mixOption(overview: Overview): ChartOption {
  const byKey = shares(overview);
  const rows = CHANNEL_STATES.filter((state) => byKey[state.key]);
  return {
    tooltip: {
      trigger: "item",
      formatter: (item: unknown) => {
        const { dataIndex, color } = item as { dataIndex: number; color: string };
        const state = rows[dataIndex];
        const share = state ? byKey[state.key] : undefined;
        if (!state || !share) return "";
        return tooltipTitle(state.label) + tooltipRow(color, formatShare(share.value), formatInterval(share.ci, "share", share.value) || "of clients");
      },
    },
    grid: { left: 0, right: MIX_RIGHT, top: 0, bottom: 0, containLabel: true },
    xAxis: { type: "value", min: 0, axisLabel: { formatter: (value: number) => formatShare(value, 0) } },
    yAxis: { type: "category", inverse: true, data: rows.map((state) => state.label), axisLine: { show: false } },
    series: [
      {
        type: "bar",
        id: "share",
        barWidth: "45%",
        data: rows.map((state) => ({ value: byKey[state.key]?.value ?? 0, itemStyle: { color: state.colour, borderRadius: [0, 4, 4, 0] } })),
      },
      intervalSeries({ id: "share-interval", intervals: rows.map((state) => byKey[state.key]?.ci ?? null) }),
    ],
  } as ChartOption;
}

/** The calendar's cell size and the chart height it needs, from the width it has. */
function calendarGeometry(overview: Overview, width: number) {
  const days = new Set(overview.daily.map((row) => row.day.slice(0, 10))).size;
  const halves = width > 0 && width < PHONE;
  const weeks = Math.ceil((halves ? days / 2 : days) / 7) + 1;
  const room = width - CALENDAR.left - CALENDAR.right;
  const cell = Math.max(CALENDAR.minCell, Math.min(CALENDAR.maxCell, Math.floor(room / weeks)));
  const rows = halves ? 2 : 1;
  const height = CALENDAR.top + rows * cell * 7 + (rows - 1) * CALENDAR.gap + CALENDAR.legend;
  return { halves, cell, height };
}

function calendarOption(overview: Overview, width: number): ChartOption {
  const byDay = new Map<string, number>();
  for (const row of overview.daily) byDay.set(row.day.slice(0, 10), (byDay.get(row.day.slice(0, 10)) ?? 0) + row.events);
  const days = [...byDay.entries()].sort(([a], [b]) => a.localeCompare(b));
  const values = days.map(([, value]) => value);
  const first = days[0]?.[0] ?? overview.period[0];
  const last = days[days.length - 1]?.[0] ?? overview.period[1];
  // A phone gets the year in two halves, so a day stays a cell one can see and touch.
  const { halves, cell } = calendarGeometry(overview, width);
  const middle = days[Math.floor(days.length / 2)]?.[0] ?? last;
  const ranges = halves ? [[first, middle], [middle, last]] : [[first, last]];
  return {
    tooltip: {
      formatter: (item: unknown) => {
        const { value, color } = item as { value: [string, number]; color: string };
        return tooltipTitle(dayFormat.format(new Date(`${value[0]}T00:00:00Z`))) + tooltipRow(color, `${formatCount(value[1])} events`, "that day");
      },
    },
    visualMap: {
      min: Math.min(...values, 0),
      max: Math.max(...values, 1),
      orient: "horizontal",
      left: CALENDAR.left,
      bottom: 0,
      itemHeight: CALENDAR.legendLength,
      itemWidth: CALENDAR.legendThickness,
      calculable: false,
      text: ["busier", "quieter"],
      inRange: { color: ["token:chart-seq-low", "token:chart-seq-high"] },
    },
    calendar: ranges.map((range, index) => ({
      range,
      top: CALENDAR.top + index * (cell * 7 + CALENDAR.gap),
      left: CALENDAR.left,
      right: CALENDAR.right,
      cellSize: ["auto", cell],
      orient: "horizontal",
      dayLabel: { firstDay: 1, nameMap: ["S", "M", "T", "W", "T", "F", "S"] },
    })),
    series: ranges.map((_, index) => ({
      type: "heatmap",
      id: `days-${index}`,
      coordinateSystem: "calendar",
      calendarIndex: index,
      data: days,
    })),
  } as ChartOption;
}

export function OverviewPage() {
  const overview = useQuery({ queryKey: ["overview"], queryFn: api.overview });
  const [calendarBox, calendarWidth] = useElementWidth<HTMLDivElement>();
  const data = overview.data;

  const mix = useMemo(() => (data ? mixOption(data) : null), [data]);
  const calendar = useMemo(() => (data ? calendarOption(data, calendarWidth) : null), [data, calendarWidth]);
  const byKey = data ? shares(data) : {};
  const intervals = data?.channel_mix_intervals;

  return (
    <>
      <PageHeader
        title="Overview"
        lead="The synthetic clients of the reference pipeline: how many there are, which channels they use, and how their activity moves through the year."
      />
      {overview.error ? <Failure error={overview.error} onRetry={() => void overview.refetch()} /> : null}
      {overview.isPending ? (
        <Loading what="the overview">
          <div className="grid gap-8 lg:grid-cols-[minmax(16rem,20rem)_minmax(0,1fr)]">
            <div className="flex flex-col gap-6">
              {FIGURES.concat(FIGURES.slice(0, 2)).map((figure, index) => (
                <div key={index} className="flex flex-col gap-2">
                  <Bone className="h-3 w-24" />
                  <Bone className="h-8 w-40" />
                  <Bone className="h-3 w-56" />
                </div>
              ))}
            </div>
            <div className="flex flex-col gap-6">
              <Bone className="h-64 w-full" />
              <Bone className="h-56 w-full" />
            </div>
          </div>
        </Loading>
      ) : null}
      {data && mix && calendar ? (
        <div className="grid gap-8 lg:grid-cols-[minmax(16rem,20rem)_minmax(0,1fr)]">
          <section aria-labelledby="overview-numbers" className="flex flex-col gap-4">
            <h2 id="overview-numbers" className="sr-only">
              The numbers
            </h2>
            <dl className="flex flex-col gap-6">
              <Figure label="Clients" value={formatCount(data.clients)} context={`synthetic clients in the client file, ${formatPeriod(data.period)}`} />
              <Figure label="Events" value={formatCount(data.events)} context="card, online and branch events over the same period" />
              {FIGURES.map((figure) => {
                const share = byKey[figure.key];
                return share ? (
                  <Figure
                    key={figure.key}
                    label={figure.label}
                    value={formatShare(share.value)}
                    interval={formatInterval(share.ci, "share", share.value) || undefined}
                    context={figure.context}
                  />
                ) : null;
              })}
            </dl>
            <p className="max-w-measure border-t border-border pt-3 text-label text-ink-secondary">
              {intervals
                ? `Shares of ${formatCount(intervals.n)} clients, each with its ${intervals.interval} from the ledger.`
                : `Shares of ${formatCount(data.clients)} clients. Their intervals appear once the synthetic ledger is built (just ledger).`}
            </p>
          </section>
          <div className="flex min-w-0 flex-col gap-6">
            <Panel title="Channel mix" subtitle={`Share of clients by the channels they used during the year${intervals ? `, with the ${intervals.interval}` : ""}.`}>
              <Chart option={mix} height="compact" label={`Bar chart of the share of clients in each channel profile: ${CHANNEL_STATES.filter((s) => byKey[s.key]).map((s) => `${stateLabel(s.key)} ${formatShare(byKey[s.key]?.value)}`).join(", ")}`} />
            </Panel>
            <Panel title="Daily activity" subtitle="Events per day across all channels; the stronger the blue, the busier the day.">
              <div ref={calendarBox}>
                <Chart option={calendar} pixels={calendarGeometry(data, calendarWidth).height} label="Calendar heatmap of events per day over the year" />
              </div>
            </Panel>
          </div>
        </div>
      ) : null}
    </>
  );
}
