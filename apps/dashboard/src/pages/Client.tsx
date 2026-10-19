import { useQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";
import { motion } from "motion/react";
import { useEffect, useId, useMemo, useState, type FormEvent, type ReactNode } from "react";
import { Link } from "react-router";

import { api, ApiError, type Client360 } from "../api";
import { CHANNEL_STATES, cssToken, stateColour, stateLabel } from "../channels";
import { tooltipRow, tooltipTitle } from "../charts/theme";
import { Chart, type ChartOption } from "../components/Chart";
import { Button } from "../components/Pressable";
import { Bone, EmptyState, Failure, Loading, Notice, PageHeader, Panel, ScrollRow, Tag } from "../components/ui";
import { featureLabel, formatCount, formatDate, formatMonth, formatNumber, formatShare, humanize } from "../format";
import { CROSS_FADE, SPRING, useCalmMotion } from "../motion";

const SECTIONS: [string, string][] = [
  ["profile", "Profile"],
  ["timeline", "Month by month"],
  ["segment", "Segment"],
  ["risk", "Silence risk"],
  ["alerts", "Open alerts"],
];

function timelineOption(client: Client360): ChartOption {
  const rows = client.timeline;
  return {
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (params: unknown) => {
        const first = (Array.isArray(params) ? params[0] : params) as { dataIndex: number; color: string };
        const row = rows[first.dataIndex];
        if (!row) return "";
        return tooltipTitle(formatMonth(row.month_start)) + tooltipRow(first.color, `${formatCount(row.n_events)} events`, stateLabel(row.channel_state));
      },
    },
    xAxis: { type: "category", data: rows.map((row) => formatMonth(row.month_start)) },
    yAxis: { type: "value", minInterval: 1, name: "events" },
    series: [
      {
        type: "bar",
        id: "events",
        data: rows.map((row) => ({ value: row.n_events, itemStyle: { color: stateColour(row.channel_state), borderRadius: [4, 4, 0, 0] } })),
      },
    ],
  } as ChartOption;
}

function Fact({ term, children }: { term: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <dt className="text-label text-ink-secondary">{term}</dt>
      <dd className="text-small font-label text-ink">{children}</dd>
    </div>
  );
}

function Profile({ client }: { client: Client360 }) {
  const p = client.profile as Record<string, unknown>;
  const number = (key: string) => (typeof p[key] === "number" ? (p[key] as number) : null);
  const text = (key: string) => (typeof p[key] === "string" ? (p[key] as string) : null);
  const age = number("age");
  const tenure = number("tenure_years");
  const lastEvent = text("last_event_date");
  const recency = number("recency_days");
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-3 xl:grid-cols-4">
      <Fact term="Client">
        <span className="font-mono">{client.client_id}</span>
      </Fact>
      <Fact term="Age">{age === null ? "not known" : `${formatCount(age)} years${text("age_band") ? ` (band ${text("age_band")})` : ""}`}</Fact>
      <Fact term="Tenure">{tenure === null ? "not known" : `${formatNumber(tenure, 1)} years`}</Fact>
      <Fact term="Channel profile">
        <span className="inline-flex items-center gap-2">
          <span aria-hidden className="size-2.5 rounded-full" style={{ background: cssToken(stateColour(String(p.channel_profile ?? "none"))) }} />
          {stateLabel(String(p.channel_profile ?? "none"))}
        </span>
      </Fact>
      <Fact term="Events in the year">
        {formatCount(number("n_events") ?? 0)}
        <span className="block text-label font-body text-ink-secondary">
          card {formatCount(number("n_card") ?? 0)} · online {formatCount(number("n_online") ?? 0)} · branch {formatCount(number("n_branch") ?? 0)}
        </span>
      </Fact>
      <Fact term="Active months">{formatCount(number("active_months") ?? 0)}</Fact>
      <Fact term="Last event">
        {lastEvent ? formatDate(lastEvent.slice(0, 10)) : "none in the year"}
        {recency === null ? null : <span className="block text-label font-body text-ink-secondary">{formatCount(recency)} days before the end of the period</span>}
      </Fact>
    </dl>
  );
}

/** The three features that push this client's risk up the most, as bars from zero. */
function Reasons({ client }: { client: Client360 }) {
  const calm = useCalmMotion();
  const reasons = client.silence_risk?.reasons ?? [];
  const largest = Math.max(...reasons.map((reason) => Math.abs(Number(reason.shap ?? 0))), Number.EPSILON);
  return (
    <ol className="flex flex-col gap-3">
      {reasons.map((reason) => {
        const shap = Number(reason.shap ?? 0);
        return (
          <li key={String(reason.feature)} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1">
            <span className="text-small text-ink">{featureLabel(String(reason.feature))}</span>
            <span className="text-small text-ink-secondary">
              {shap >= 0 ? "+" : "-"}
              {formatNumber(Math.abs(shap), 2)} log-odds
            </span>
            <span className="col-span-2 h-1.5 overflow-hidden rounded-full bg-fill-control" aria-hidden>
              <motion.span
                className="block h-full origin-left rounded-full bg-accent"
                initial={calm ? { opacity: 0, scaleX: Math.abs(shap) / largest } : { scaleX: 0 }}
                animate={{ opacity: 1, scaleX: Math.abs(shap) / largest }}
                transition={calm ? CROSS_FADE : SPRING}
              />
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function jump(id: string, calm: boolean) {
  const section = document.getElementById(`client-${id}`);
  if (!section) return;
  section.scrollIntoView({ behavior: calm ? "auto" : "smooth", block: "start" });
  section.querySelector<HTMLElement>("h2")?.focus({ preventScroll: true });
}

export function ClientPage() {
  const calm = useCalmMotion();
  const searchId = useId();
  const risk = useQuery({ queryKey: ["silence", 50], queryFn: () => api.silence(50) });
  const segments = useQuery({ queryKey: ["segments"], queryFn: api.segments });
  const [chosen, setChosen] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const clientId = chosen ?? risk.data?.clients[0]?.client_id ?? null;
  const client = useQuery({
    queryKey: ["client", clientId],
    queryFn: () => api.client(clientId ?? ""),
    enabled: clientId !== null,
    retry: (count, error) => !(error instanceof ApiError && error.status === 404) && count < 1,
  });

  useEffect(() => {
    if (clientId && draft === "") setDraft(clientId);
  }, [clientId, draft]);

  const timeline = useMemo(() => (client.data ? timelineOption(client.data) : null), [client.data]);
  const segment = segments.data?.segments.find((item) => item.segment === client.data?.segment);
  const notFound = client.error instanceof ApiError && client.error.status === 404;

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const id = draft.trim();
    if (id) setChosen(id);
  };

  return (
    <>
      <PageHeader
        title="Customer profile"
        lead="One synthetic client at a time: profile, month-by-month channel use, segment, silence risk with its three reasons, and open alerts."
      />
      <form role="search" onSubmit={submit} className="mb-6 flex max-w-xl flex-col gap-2">
        <label htmlFor={searchId} className="text-small font-label text-ink">
          Client id
        </label>
        <div className="flex gap-2">
          <input
            id={searchId}
            type="search"
            list={`${searchId}-suggestions`}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            autoComplete="off"
            spellCheck={false}
            className="min-h-11 min-w-0 flex-1 rounded-control border border-border-strong bg-surface px-3 font-mono text-small text-ink sm:min-h-9"
          />
          <Button tone="primary" type="submit" icon={Search}>
            Show
          </Button>
        </div>
        <datalist id={`${searchId}-suggestions`}>
          {(risk.data?.clients ?? []).map((row) => (
            <option key={row.client_id} value={row.client_id}>
              {formatShare(row.probability)} silence risk
            </option>
          ))}
        </datalist>
        <p className="text-label text-ink-secondary">
          {risk.data
            ? `Suggestions are the ${formatCount(risk.data.clients.length)} highest silence risks; any synthetic client id works.`
            : "Type a synthetic client id."}
        </p>
      </form>
      {risk.error ? <Failure error={risk.error} onRetry={() => void risk.refetch()} /> : null}
      {notFound ? (
        <Notice kind="error" title={`No client ${chosen ?? ""} in this workspace`}>
          Check the id, or choose one of the suggestions in the search field.
        </Notice>
      ) : client.error ? (
        <Failure error={client.error} onRetry={() => void client.refetch()} />
      ) : null}
      {client.isPending && clientId ? (
        <Loading what="the client">
          <Bone className="h-8 w-full max-w-lg" />
          {[0, 1, 2].map((index) => (
            <div key={index} className="flex flex-col gap-3 rounded-panel border border-border bg-surface p-5">
              <Bone className="h-5 w-40" />
              <Bone className="h-24 w-full" />
            </div>
          ))}
        </Loading>
      ) : null}
      {client.data && timeline ? (
        <div className="flex flex-col gap-5">
          <nav aria-label="Sections of this profile">
            <ScrollRow>
              <ul className="flex gap-1 whitespace-nowrap">
              {SECTIONS.map(([id, text]) => (
                <li key={id}>
                  <a
                    href={`#client-${id}`}
                    onClick={(event) => {
                      event.preventDefault();
                      jump(id, calm);
                    }}
                    className="flex min-h-11 items-center rounded-control px-3 text-small text-ink-secondary no-underline hover:bg-fill-subtle hover:text-ink active:bg-fill-control sm:min-h-8"
                  >
                    {text}
                  </a>
                </li>
              ))}
              </ul>
            </ScrollRow>
          </nav>
          <Panel id="client-profile" title="Profile">
            <Profile client={client.data} />
          </Panel>
          <Panel id="client-timeline" title="Month by month" subtitle="Events per month, coloured by the month's channel state.">
            <ul className="mb-3 flex flex-wrap gap-x-4 gap-y-1" aria-label="Channel states">
              {CHANNEL_STATES.map((state) => (
                <li key={state.key} className="inline-flex items-center gap-2 text-label text-ink-secondary">
                  <span aria-hidden className="size-2.5 rounded-full" style={{ background: cssToken(state.colour) }} />
                  {state.label}
                </li>
              ))}
            </ul>
            <Chart option={timeline} height="compact" label="Bar chart of the client's events per month, coloured by the month's channel state" />
          </Panel>
          <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
            <Panel id="client-segment" title="Segment">
              {client.data.segment === null ? (
                <p className="text-small text-ink-secondary">Not segmented: the segments cover active clients only.</p>
              ) : (
                <div className="flex flex-col gap-2">
                  <p className="text-body font-label text-ink">
                    Segment {client.data.segment}
                    {segment ? `: ${segment.name}` : ""}
                  </p>
                  {segment ? (
                    <p className="text-small text-ink-secondary">
                      {formatShare(segment.share)} of segmented clients (n = {formatCount(segment.n_clients)}).
                    </p>
                  ) : null}
                  <Link to="/segments" className="text-small">
                    Compare it with the other segments
                  </Link>
                </div>
              )}
            </Panel>
            <Panel id="client-risk" title="Silence risk">
              {client.data.silence_risk ? (
                <div className="flex flex-col gap-4">
                  <div className="flex flex-col gap-1">
                    <p className="text-figure font-title text-ink">{formatShare(client.data.silence_risk.probability)}</p>
                    <p className="max-w-measure text-small text-ink-secondary">
                      Calibrated probability of no event in the three months after the cutoff of{" "}
                      {formatDate(client.data.silence_risk.cutoff_date.slice(0, 10))}.
                    </p>
                  </div>
                  <div className="flex flex-col gap-2">
                    <p className="text-label font-label text-ink-secondary">What pushes it up the most (SHAP, in log-odds)</p>
                    <Reasons client={client.data} />
                  </div>
                </div>
              ) : (
                <p className="text-small text-ink-secondary">Not in the scored population at the cutoff.</p>
              )}
            </Panel>
          </div>
          <Panel id="client-alerts" title="Open alerts" subtitle="Triage reasons for a person to review, never an accusation.">
            {client.data.open_alerts.length ? (
              <ul className="flex flex-col divide-y divide-border">
                {client.data.open_alerts.map((alert) => (
                  <li key={alert.id} className="flex flex-wrap items-center gap-x-4 gap-y-2 py-3 first:pt-0 last:pb-0">
                    <span className="text-small font-label text-ink">Rank {formatCount(alert.rank)}</span>
                    <span className="text-small text-ink-secondary">score {formatNumber(alert.score, 3)}</span>
                    <span className="flex flex-wrap gap-1.5">
                      {alert.rules.map((rule) => (
                        <Tag key={rule}>{humanize(rule)}</Tag>
                      ))}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <EmptyState title="No open alert for this client.">
                Alerts appear here when the monitoring layer ranks this client in its queue. The whole queue is on the{" "}
                <Link to="/alerts">Alerts</Link> page.
              </EmptyState>
            )}
          </Panel>
        </div>
      ) : null}
    </>
  );
}
