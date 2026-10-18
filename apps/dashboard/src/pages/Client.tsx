import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { api } from "../api";
import { Chart, type ChartOption } from "../components/Chart";
import { Card, Failure, Kpi, Loading, PageHeader, Tag } from "../components/ui";
import { formatCount, formatNumber, formatShare, humanize } from "../format";

const STATES = ["none", "branch_only", "digital_only", "mixed"];

export function ClientPage() {
  const risk = useQuery({ queryKey: ["silence", 50], queryFn: () => api.silence(50) });
  const [chosen, setChosen] = useState<string | null>(null);
  const clientId = chosen ?? risk.data?.clients[0]?.client_id ?? null;
  const client = useQuery({
    queryKey: ["client", clientId],
    queryFn: () => api.client(clientId ?? ""),
    enabled: clientId !== null,
  });

  const timeline = useMemo<ChartOption>(() => {
    const rows = client.data?.timeline ?? [];
    return {
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: rows.map((row) => row.month_start.slice(0, 7)) },
      yAxis: [
        { type: "value", name: "events" },
        { type: "category", data: STATES.map(humanize), show: false },
      ],
      series: [
        { type: "bar", name: "events", data: rows.map((row) => row.n_events), barMaxWidth: 28 },
        {
          type: "line",
          name: "channel state",
          yAxisIndex: 1,
          step: "middle",
          data: rows.map((row) => humanize(row.channel_state)),
        },
      ],
      legend: { top: 0 },
    };
  }, [client.data]);

  const profile = client.data?.profile ?? {};
  return (
    <>
      <PageHeader
        title="Customer 360"
        lead="One synthetic client: profile, month-by-month channel use, segment, silence risk with the reasons behind it, and open alerts."
      />
      {risk.error ? <Failure error={risk.error} /> : null}
      <label className="mb-4 flex max-w-md flex-col gap-1 text-sm">
        <span className="font-medium">Client (the fifty highest silence risks)</span>
        <select
          className="rounded-md border border-border bg-surface px-3 py-2 text-ink"
          value={clientId ?? ""}
          onChange={(event) => setChosen(event.target.value)}
        >
          {(risk.data?.clients ?? []).map((row) => (
            <option key={row.client_id} value={row.client_id}>
              {row.client_id} ({formatShare(row.probability)})
            </option>
          ))}
        </select>
      </label>
      {client.isPending && clientId ? <Loading what="the client" /> : null}
      {client.error ? <Failure error={client.error} /> : null}
      {client.data ? (
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Kpi label="Age" value={String(profile.age ?? "not known")} />
            <Kpi label="Channel profile" value={humanize(String(profile.channel_profile ?? "none"))} />
            <Kpi label="Events in the year" value={formatCount(Number(profile.n_events ?? 0))} />
            <Kpi label="Segment" value={client.data.segment === null ? "not segmented" : `#${client.data.segment}`} />
          </div>
          <Card title="Month by month" subtitle="Events per month and the month's channel state.">
            <Chart option={timeline} label="Bar chart of the client's monthly events with the channel state" height={260} />
          </Card>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <Card title="Silence risk" subtitle="Calibrated probability of no event in the next three months.">
              {client.data.silence_risk ? (
                <>
                  <p className="text-3xl font-semibold tabular-nums">{formatShare(client.data.silence_risk.probability)}</p>
                  <ul className="mt-3 flex flex-col gap-1 text-sm">
                    {client.data.silence_risk.reasons.map((reason) => (
                      <li key={String(reason.feature)}>
                        {humanize(String(reason.feature))}{" "}
                        <span className="text-ink-muted">(SHAP {formatNumber(Number(reason.shap), 2)})</span>
                      </li>
                    ))}
                  </ul>
                </>
              ) : (
                <p className="text-sm text-ink-muted">Not in the scored population.</p>
              )}
            </Card>
            <Card title="Open alerts" subtitle="Triage reasons, for a person to review.">
              {client.data.open_alerts.length ? (
                <ul className="flex flex-col gap-2">
                  {client.data.open_alerts.map((alert) => (
                    <li key={alert.id} className="flex flex-wrap gap-2">
                      <Tag>rank {alert.rank}</Tag>
                      {alert.rules.map((rule) => (
                        <Tag key={rule}>{humanize(rule)}</Tag>
                      ))}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-ink-muted">No open alert.</p>
              )}
            </Card>
          </div>
        </div>
      ) : null}
    </>
  );
}
