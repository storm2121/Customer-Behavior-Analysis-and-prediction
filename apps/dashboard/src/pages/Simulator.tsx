import { keepPreviousData, useQueries, useQuery, useQueryClient, type QueryKey } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { api, type Bands, type Simulation, type SimulationRequest } from "../api";
import { tooltipRow, tooltipTitle } from "../charts/theme";
import { Chart, type ChartOption } from "../components/Chart";
import { Segmented } from "../components/Segmented";
import { Slider } from "../components/Slider";
import { Bone, Failure, Loading, Notice, PageHeader, Panel, Tag } from "../components/ui";
import { formatCount, formatEstimate, formatShare } from "../format";
import { useDebounced } from "../hooks";
import { bandsAt, computedPolicies } from "../simulator";

// The policy's controls. Ages are the upper edges of the age bands the simulator moves.
const SHARE = { min: 0, max: 1, step: 0.05 };
const AGE = { min: 25, max: 65, step: 10 };
const AGES = [25, 35, 45, 55, 65];
const HORIZON = 12;
const SAMPLES = 600;
// The network waits for the slider to rest this long; the visual never waits.
const SETTLE_MS = 250;

type Target = SimulationRequest["target"];

const TARGETS: { value: Target; label: string; words: string }[] = [
  { value: "mixed", label: "Mixed use", words: "mixed use" },
  { value: "digital_only", label: "Digital only", words: "digital only" },
];

function request(share: number, maxAge: number, target: Target): SimulationRequest {
  return { share, max_age: maxAge, target, horizon: HORIZON, samples: SAMPLES };
}

function key(body: SimulationRequest): QueryKey {
  return ["simulate", body];
}

/**
 * A fan chart: the posterior median of `quantity` for the policy and the baseline, each with its
 * band (drawn as a stacked area from the low to the high percentile). The policy is the thing
 * being discussed, in the accent; the baseline is in the neutral ramp.
 */
function fan(months: number[], policy: Bands, baseline: Bands, quantity: string, unit: "share" | "count", interval: string): ChartOption {
  const format = (value: number | undefined) => (value === undefined ? "not defined" : unit === "share" ? formatShare(value) : formatCount(value));
  const band = (bands: Bands, name: string, colour: string) => {
    const q = bands[quantity];
    const low = q?.p05 ?? [];
    return [
      { type: "line" as const, id: `${name}-low`, data: low, stack: name, symbol: "none", lineStyle: { opacity: 0 }, silent: true, tooltip: { show: false } },
      {
        type: "line" as const,
        id: `${name}-band`,
        data: (q?.p95 ?? []).map((high, index) => high - (low[index] ?? high)),
        stack: name,
        symbol: "none",
        lineStyle: { opacity: 0 },
        areaStyle: { color: colour, opacity: 0.14 },
        silent: true,
        tooltip: { show: false },
      },
      { type: "line" as const, id: name, name: name === "policy" ? "With the policy" : "Without it", data: q?.p50 ?? [], symbol: "none", itemStyle: { color: colour }, lineStyle: { color: colour, width: 2 } },
    ];
  };
  const row = (bands: Bands, words: string, index: number, colour: string) => {
    const q = bands[quantity];
    const median = q?.p50[index];
    return tooltipRow(colour, median === undefined ? "not defined" : formatEstimate(median, [q?.p05[index] ?? null, q?.p95[index] ?? null], unit), words);
  };
  return {
    tooltip: {
      trigger: "axis",
      formatter: (params: unknown) => {
        const items = (Array.isArray(params) ? params : [params]) as { dataIndex?: number; seriesId?: string; color?: string }[];
        const index = items[0]?.dataIndex ?? 0;
        const colour = (id: string) => items.find((item) => item.seriesId === id)?.color ?? "";
        return [
          tooltipTitle(`Month ${months[index] ?? index}, ${interval}`),
          row(policy, "with the policy", index, colour("policy")),
          row(baseline, "without it", index, colour("baseline")),
        ].join("");
      },
    },
    legend: { top: 0, left: 0, data: ["With the policy", "Without it"] },
    xAxis: { type: "category", data: months.map((month) => `${month}`), name: "months from now", nameLocation: "middle", nameGap: 28, boundaryGap: false },
    yAxis: { type: "value", scale: true, axisLabel: { formatter: (value: number) => format(value) } },
    series: [...band(baseline, "baseline", "token:chart-neutral"), ...band(policy, "policy", "token:accent")],
  } as ChartOption;
}

function last(values: number[] | undefined): number | undefined {
  return values && values.length ? values[values.length - 1] : undefined;
}

function Outcome({ title, unit, policy, baseline, quantity }: { title: string; unit: "share" | "count"; policy: Bands; baseline: Bands; quantity: string }) {
  const read = (bands: Bands) => {
    const q = bands[quantity];
    const median = last(q?.p50);
    return median === undefined ? "not defined" : formatEstimate(median, [last(q?.p05) ?? null, last(q?.p95) ?? null], unit);
  };
  return (
    <div className="flex flex-col gap-1">
      <dt className="text-label font-label text-ink-secondary">{title}</dt>
      <dd className="flex flex-col gap-0.5 text-small">
        <span className="text-ink">
          <span className="font-label">{read(policy)}</span> with the policy
        </span>
        <span className="text-ink-secondary">{read(baseline)} without it</span>
      </dd>
    </div>
  );
}

export function SimulatorPage() {
  const [share, setShare] = useState(0.3);
  const [maxAge, setMaxAge] = useState(45);
  const [target, setTarget] = useState<Target>("mixed");
  const [liveShare, setLiveShare] = useState<number | null>(null);
  const [liveAge, setLiveAge] = useState<number | null>(null);
  const [ageIntent, setAgeIntent] = useState(false);

  // What the network computes: the settled policy, debounced.
  const settled = useDebounced(request(share, maxAge, target), SETTLE_MS);
  const exact = useQuery({ queryKey: key(settled), queryFn: () => api.simulate(settled), placeholderData: keepPreviousData });

  // What the screen shows: the share thumb's own value at every frame. The age limit moves in
  // whole bands, so its bands switch as the thumb crosses each step (onChange), not in between.
  const shownShare = liveShare ?? share;
  const shownAge = maxAge;

  // Anchors for the interpolation: the whole eligible group moved, at each age limit. The current
  // age's anchor follows the first answer (so the first request is the policy on screen); the
  // others load when the age slider is first touched.
  useQueries({
    queries: AGES.map((age) => {
      const body = request(1, age, target);
      return {
        queryKey: key(body),
        queryFn: () => api.simulate(body),
        enabled: exact.isSuccess && (age === shownAge || ageIntent),
        staleTime: Number.POSITIVE_INFINITY,
      };
    }),
  });

  // Every policy computed so far for the age and target on screen: the query cache keeps each
  // settled answer and each anchor, so the interpolation gets finer as the reader explores.
  const queryClient = useQueryClient();
  const group = queryClient
    .getQueriesData<Simulation>({ queryKey: ["simulate"] })
    .flatMap(([queryKey, data]) => {
      const body = queryKey[1] as SimulationRequest | undefined;
      if (!data || !body || body.max_age !== shownAge || body.target !== target || body.horizon !== HORIZON || body.samples !== SAMPLES) return [];
      return [{ share: body.share, policy: data.policy_bands, baseline: data.baseline_bands, eligible: data.eligible_bands }];
    });
  const points = computedPolicies(group);
  const interpolated = bandsAt(points, shownShare);
  const baseline = group[0]?.baseline ?? exact.data?.baseline_bands ?? null;
  const policy = interpolated ?? exact.data?.policy_bands ?? null;
  const eligible = group[0]?.eligible ?? exact.data?.eligible_bands ?? [];
  const months = exact.data?.months ?? [];
  const interval = exact.data?.interval ?? "band";
  const moving = liveShare !== null || liveAge !== null;
  // The bands are the API's own only when its answer is for exactly the controls on screen and no
  // thumb is moving; until then they are an interpolation, drawn faded and marked as updating, so
  // nobody reads one as a computed band.
  const answered =
    exact.data !== undefined &&
    !exact.isPlaceholderData &&
    !moving &&
    settled.share === share &&
    settled.max_age === maxAge &&
    settled.target === target;
  const provisional = !answered;
  const fade = `transition-opacity duration-[var(--duration-fade)] ${provisional ? "opacity-50" : "opacity-100"}`;

  const charts = useMemo(() => {
    if (!policy || !baseline || months.length === 0) return null;
    return {
      digital: fan(months, policy, baseline, "digital_share", "share", interval),
      branch: fan(months, policy, baseline, "branch_load", "count", interval),
    };
  }, [policy, baseline, months, interval]);

  const targetWords = TARGETS.find((item) => item.value === target)?.words ?? target;
  const horizon = months.length ? months[months.length - 1] : HORIZON;

  return (
    <>
      <PageHeader
        title="Simulator"
        lead="A Markov chain of monthly channel states with a Dirichlet posterior. Move a share of the branch-only clients under an age limit to another state this month, and see the year that follows with bands from the posterior."
      />
      <div className="grid items-start gap-6 lg:grid-cols-[minmax(17rem,22rem)_minmax(0,1fr)]">
        <Panel title="Policy" className="lg:sticky lg:top-[calc(var(--bar-height)+var(--space-unit)*4)]">
          <div className="flex flex-col gap-5">
            <Slider
              label="Share of eligible clients who move"
              value={share}
              {...SHARE}
              format={(value) => formatShare(value, 0)}
              onChange={setShare}
              onLive={setLiveShare}
            />
            <Slider
              label="Clients younger than"
              value={maxAge}
              {...AGE}
              format={(value) => `${formatCount(value)} years`}
              onChange={setMaxAge}
              onLive={setLiveAge}
              onIntent={() => setAgeIntent(true)}
            />
            <Segmented label="They move to" options={TARGETS.map(({ value, label }) => ({ value, label }))} value={target} onChange={setTarget} />
            <div className="flex flex-wrap gap-2">
              <Tag>Eligible age bands: {eligible.join(", ") || "none"}</Tag>
              <Tag>A what-if under the fitted chain, not a forecast</Tag>
            </div>
          </div>
        </Panel>
        <div className="flex min-w-0 flex-col gap-5">
          <div className="flex flex-col gap-1">
            <p className="text-heading font-title text-ink">
              Move {formatShare(share, 0)} of branch-only clients under {formatCount(maxAge)} to {targetWords}
            </p>
            <p aria-live="polite" className="min-h-5 text-label text-ink-secondary">
              {charts && provisional ? "Updating: these bands are interpolated until the API answers." : null}
            </p>
          </div>
          {exact.error ? <Failure error={exact.error} onRetry={() => void exact.refetch()} /> : null}
          {exact.isPending ? (
            <Loading what="the simulation">
              <Bone className="h-16 w-full" />
              <Bone className="h-72 w-full" />
              <Bone className="h-72 w-full" />
            </Loading>
          ) : null}
          {charts && policy && baseline ? (
            <>
              <dl className={`grid grid-cols-1 gap-4 sm:grid-cols-2 ${fade}`}>
                <Outcome title={`Digital share in month ${horizon}`} unit="share" quantity="digital_share" policy={policy} baseline={baseline} />
                <Outcome title={`Branch operations in month ${horizon}`} unit="count" quantity="branch_load" policy={policy} baseline={baseline} />
              </dl>
              <div className="flex flex-col gap-5">
                <Panel title="Digital share" subtitle={`Clients using card or online banking in the month: posterior median and ${interval}.`}>
                  <div className={fade}>
                    <Chart option={charts.digital} transition={moving ? "immediate" : "smooth"} label="Digital share over the coming months, with the policy against without it, with bands" />
                  </div>
                </Panel>
                <Panel title="Branch operations" subtitle={`Expected branch operations in the month: posterior median and ${interval}.`}>
                  <div className={fade}>
                    <Chart option={charts.branch} transition={moving ? "immediate" : "smooth"} label="Branch operations over the coming months, with the policy against without it, with bands" />
                  </div>
                </Panel>
              </div>
              <Notice kind="status">
                While a slider moves, the bands are interpolated between policies the API has already computed, drawn faded and marked as updating; once the API answers for the value you stop at, the bands are its own.
              </Notice>
            </>
          ) : null}
        </div>
      </div>
    </>
  );
}
