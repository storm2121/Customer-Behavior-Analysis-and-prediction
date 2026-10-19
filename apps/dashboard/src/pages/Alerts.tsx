import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Ban, Check, CircleQuestionMark, Flag, PanelRightOpen, type LucideIcon } from "lucide-react";
import { useRef, useState } from "react";

import { api, type Alert, type Reason } from "../api";
import { Drawer } from "../components/Drawer";
import { Icon } from "../components/Icon";
import { Button } from "../components/Pressable";
import { Segmented } from "../components/Segmented";
import { Bone, EmptyState, Failure, Loading, Notice, PageHeader, Tag } from "../components/ui";
import { formatCount, formatDate, formatNumber, formatValue, humanize, label } from "../format";

const BUDGETS = [10, 25, 50];

const REVIEWS: { label: string; words: string; done: string; icon: LucideIcon }[] = [
  { label: "confirmed", words: "Worth escalating", done: "Marked worth escalating", icon: Flag },
  { label: "dismissed", words: "Dismiss", done: "Dismissed", icon: Ban },
  { label: "needs_information", words: "Needs information", done: "Marked as needing information", icon: CircleQuestionMark },
];

// The unit of each piece of evidence the triage rules and scores record (strata.models.anomaly).
const EVIDENCE_UNITS: Record<string, string> = {
  deposits: "count",
  events: "count",
  amounts: "count",
  largest: "amount",
  threshold: "amount",
  total: "amount",
  deposit: "amount",
  withdrawal: "amount",
  branch_deposits: "amount",
  card_spend: "amount",
  amount: "amount",
  window_days: "days",
  ratio: "ratio",
  robust_z: "number",
  mad: "number",
};

const DATE = /^\d{4}-\d{2}-\d{2}/;

function evidenceValue(key: string, value: unknown): string {
  if (typeof value === "number") return formatValue(value, EVIDENCE_UNITS[key]);
  if (typeof value === "string" && DATE.test(value)) return formatDate(value.slice(0, 10));
  return String(value);
}

function reasonTitle(reason: Reason): string {
  if (reason.source === "rule") return label(String(reason.rule));
  return label(String(reason.detail ?? reason.source));
}

/** The facts behind one reason: a rule's evidence, or a score's own fields. */
function evidenceOf(reason: Reason): [string, unknown][] {
  if (reason.source === "rule") return Object.entries(reason.evidence ?? {});
  return Object.entries(reason).filter(([key]) => !["source", "detail", "rule", "evidence"].includes(key));
}

function reviewOf(alert: Alert) {
  return REVIEWS.find((review) => review.label === alert.label) ?? null;
}

function ReviewState({ alert }: { alert: Alert }) {
  const review = reviewOf(alert);
  if (!review) return <span className="text-label text-ink-secondary">Not reviewed</span>;
  return (
    <span className="inline-flex items-center gap-1.5 text-label font-label text-ink">
      <Icon icon={review.icon} />
      {review.done}
    </span>
  );
}

/** Feedback buttons that change state where they are: pressed, then stored, or an error in place. */
function Feedback({ alert, budget }: { alert: Alert; budget: number }) {
  const client = useQueryClient();
  const feedback = useMutation({
    mutationFn: (value: string) => api.feedback(alert.id, value, "dashboard"),
    onSuccess: () => client.invalidateQueries({ queryKey: ["alerts", budget] }),
  });
  const chosen = feedback.isPending ? feedback.variables : (feedback.data ? feedback.variables : alert.label);
  const stored = REVIEWS.find((review) => review.label === chosen);
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-2" role="group" aria-label={`Feedback on alert ${alert.rank}`}>
        {REVIEWS.map((review) => {
          const pressed = chosen === review.label;
          return (
            <Button
              key={review.label}
              tone={pressed ? "primary" : "secondary"}
              icon={pressed && !feedback.isPending ? Check : review.icon}
              aria-pressed={pressed}
              disabled={feedback.isPending}
              onClick={() => feedback.mutate(review.label)}
            >
              {review.words}
            </Button>
          );
        })}
      </div>
      <div aria-live="polite">
        {feedback.isPending ? <p className="text-small text-ink-secondary">Storing your review…</p> : null}
        {feedback.isSuccess && stored ? (
          <Notice kind="completion">{stored.done}. The review is stored with the alert and counts towards the queue's precision.</Notice>
        ) : null}
      </div>
      {feedback.error ? <Failure error={feedback.error} onRetry={() => feedback.variables && feedback.mutate(feedback.variables)} /> : null}
    </div>
  );
}

function Evidence({ alert, budget }: { alert: Alert; budget: number }) {
  return (
    <div className="flex flex-col gap-5">
      <section aria-labelledby={`alert-${alert.id}-reasons`} className="flex flex-col gap-3">
        <h3 id={`alert-${alert.id}-reasons`} className="text-label font-label text-ink-secondary">
          Why it is in the queue
        </h3>
        <ul className="flex flex-col gap-3">
          {alert.reasons.map((reason, index) => {
            const facts = evidenceOf(reason);
            return (
              <li key={`${alert.id}-${index}`} className="rounded-panel border border-border bg-surface p-3">
                <p className="text-small font-label text-ink">{reasonTitle(reason)}</p>
                {reason.source !== "rule" ? <p className="text-label text-ink-secondary">{humanize(String(reason.source))}</p> : null}
                {facts.length ? (
                  <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1.5">
                    {facts.map(([key, value]) => (
                      <div key={key} className="flex min-w-0 flex-col">
                        <dt className="text-label text-ink-secondary">{label(key)}</dt>
                        <dd className="text-small text-ink">{evidenceValue(key, value)}</dd>
                      </div>
                    ))}
                  </dl>
                ) : null}
              </li>
            );
          })}
        </ul>
      </section>
      <section aria-labelledby={`alert-${alert.id}-review`} className="flex flex-col gap-3">
        <h3 id={`alert-${alert.id}-review`} className="text-label font-label text-ink-secondary">
          Your review
        </h3>
        <Feedback alert={alert} budget={budget} />
        <p className="text-label text-ink-secondary">Triage, never an accusation: a person decides what an alert means.</p>
      </section>
    </div>
  );
}

export function AlertsPage() {
  const [budget, setBudget] = useState(10);
  const [open, setOpen] = useState<number | null>(null);
  const alerts = useQuery({ queryKey: ["alerts", budget], queryFn: () => api.alerts(budget), placeholderData: (previous) => previous });
  const list = alerts.data?.alerts ?? [];
  const selected = list.find((alert) => alert.id === open) ?? null;
  // The drawer keeps showing the last alert while it slides out, then lets it go.
  const last = useRef<Alert | null>(null);
  if (selected) last.current = selected;
  const shown = selected ?? last.current;

  return (
    <>
      <PageHeader
        title={alerts.data ? `Alerts (${formatCount(list.length)} ranked)` : "Alerts"}
        lead="The monitoring layer's queue: clients ranked for a person to look at, each with the evidence a reviewer needs. Triage, never an accusation."
      />
      <div className="mb-5">
        <Segmented
          label="Alert budget"
          options={BUDGETS.map((value) => ({ value, label: formatCount(value) }))}
          value={budget}
          onChange={setBudget}
        />
      </div>
      {alerts.error ? <Failure error={alerts.error} onRetry={() => void alerts.refetch()} /> : null}
      {alerts.isPending ? (
        <Loading what="the queue">
          {[0, 1, 2, 3, 4].map((index) => (
            <Bone key={index} className="h-14 w-full" />
          ))}
        </Loading>
      ) : null}
      {alerts.data && list.length === 0 ? (
        <EmptyState title="The queue is empty." command="just anomaly">
          Ranked alerts appear here once the monitoring layer has run on this workspace.
        </EmptyState>
      ) : null}
      {list.length ? (
        <ol className={`flex flex-col rounded-panel border border-border bg-surface ${alerts.isPlaceholderData ? "opacity-60" : ""}`}>
          {list.map((alert) => (
            <li
              key={alert.id}
              className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-x-4 gap-y-2 border-b border-border px-4 py-3 last:border-b-0 sm:grid-cols-[3rem_minmax(8rem,12rem)_6rem_minmax(0,1fr)_12rem_auto] sm:py-1.5"
            >
              <span className="text-small font-title text-ink">{formatCount(alert.rank)}</span>
              <span className="truncate font-mono text-label text-ink">{alert.client_id}</span>
              <span className="hidden text-small text-ink-secondary sm:inline">score {formatNumber(alert.score, 3)}</span>
              <span className="col-span-3 flex flex-wrap gap-1.5 sm:col-span-1">
                {alert.rules.map((rule) => (
                  <Tag key={rule}>{humanize(rule)}</Tag>
                ))}
              </span>
              <span className="col-span-2 sm:col-span-1">
                <ReviewState alert={alert} />
              </span>
              <Button tone="quiet" icon={PanelRightOpen} aria-label={`Evidence for alert ${alert.rank}`} onClick={() => setOpen(alert.id)}>
                Evidence
              </Button>
            </li>
          ))}
        </ol>
      ) : null}
      <Drawer
        open={selected !== null}
        onOpenChange={(value) => (value ? null : setOpen(null))}
        title={shown ? `Alert ${formatCount(shown.rank)}` : "Alert"}
        description={shown ? `Client ${shown.client_id}, combined score ${formatNumber(shown.score, 3)}` : ""}
      >
        {shown ? <Evidence key={shown.id} alert={shown} budget={budget} /> : null}
      </Drawer>
    </>
  );
}
