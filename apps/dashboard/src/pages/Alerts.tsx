import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, type Alert, type Reason } from "../api";
import { Card, Failure, Loading, PageHeader, Tag } from "../components/ui";
import { formatNumber, humanize } from "../format";

const BUDGETS = [10, 25, 50];
const LABELS: { label: string; words: string }[] = [
  { label: "confirmed", words: "Worth escalating" },
  { label: "dismissed", words: "Dismiss" },
  { label: "needs_information", words: "Needs information" },
];

function reasonText(reason: Reason): string {
  if (reason.source === "rule") return humanize(String(reason.rule));
  return String(reason.detail ?? humanize(String(reason.source)));
}

function AlertRow({ alert, budget }: { alert: Alert; budget: number }) {
  const client = useQueryClient();
  const feedback = useMutation({
    mutationFn: (label: string) => api.feedback(alert.id, label, "dashboard"),
    onSuccess: () => client.invalidateQueries({ queryKey: ["alerts", budget] }),
  });
  return (
    <li className="flex flex-col gap-2 border-t border-border py-3 first:border-t-0">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-semibold tabular-nums">#{alert.rank}</span>
        <span className="font-mono text-xs">{alert.client_id}</span>
        <Tag>score {formatNumber(alert.score, 3)}</Tag>
        {alert.label ? <Tag>reviewed: {humanize(alert.label)}</Tag> : null}
      </div>
      <ul className="flex flex-wrap gap-2">
        {alert.reasons.map((reason, index) => (
          <li key={`${alert.id}-${index}`}>
            <Tag>{reasonText(reason)}</Tag>
          </li>
        ))}
      </ul>
      <div className="flex flex-wrap gap-2" role="group" aria-label={`Feedback on alert ${alert.rank}`}>
        {LABELS.map(({ label, words }) => (
          <button
            key={label}
            type="button"
            disabled={feedback.isPending}
            onClick={() => feedback.mutate(label)}
            className="rounded-md border border-border bg-surface px-3 py-1 text-xs text-ink hover:bg-surface-muted disabled:opacity-50"
          >
            {words}
          </button>
        ))}
      </div>
      {feedback.error ? <Failure error={feedback.error} /> : null}
    </li>
  );
}

export function AlertsPage() {
  const [budget, setBudget] = useState(10);
  const alerts = useQuery({ queryKey: ["alerts", budget], queryFn: () => api.alerts(budget) });
  return (
    <>
      <PageHeader
        title="Alerts"
        lead="The monitoring layer's queue: clients ranked for a person to look at, each with the reasons a reviewer needs. Triage, never an accusation."
      />
      <div className="mb-4 flex flex-wrap items-center gap-2" role="group" aria-label="Alert budget">
        <span className="text-sm text-ink-muted">Alert budget</span>
        {BUDGETS.map((value) => (
          <button
            key={value}
            type="button"
            aria-pressed={budget === value}
            onClick={() => setBudget(value)}
            className={`rounded-md border px-3 py-1 text-sm ${
              budget === value ? "border-accent bg-accent text-accent-ink" : "border-border bg-surface text-ink"
            }`}
          >
            {value}
          </button>
        ))}
      </div>
      {alerts.isPending ? <Loading what="the queue" /> : null}
      {alerts.error ? <Failure error={alerts.error} /> : null}
      {alerts.data ? (
        <Card title="Queue" subtitle="Feedback is stored with the alert and feeds the queue's precision.">
          <ul>
            {alerts.data.alerts.map((alert) => (
              <AlertRow key={alert.id} alert={alert} budget={budget} />
            ))}
          </ul>
        </Card>
      ) : null}
    </>
  );
}
