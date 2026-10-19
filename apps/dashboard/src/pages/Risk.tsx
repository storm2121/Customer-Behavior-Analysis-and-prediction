import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { api, type SilenceEvaluation } from "../api";
import { intervalSeries, referenceLine } from "../charts/marks";
import { tooltipRow, tooltipTitle } from "../charts/theme";
import { Chart, type ChartOption } from "../components/Chart";
import { Bone, EmptyState, Failure, Loading, PageHeader, Panel, ScrollArea, Tag } from "../components/ui";
import { featureLabel, formatCount, formatDate, formatFixed, formatShare } from "../format";

const SERVED = "lightgbm_platt";
const TOP = 25;

const MODELS: { key: string; name: string }[] = [
  { key: "majority", name: "Majority class" },
  { key: "activity_rule", name: "Activity rule" },
  { key: "glm", name: "Logistic GLM" },
  { key: "lightgbm", name: "Gradient-boosted trees" },
  { key: "lightgbm_isotonic", name: "Trees, isotonic calibration" },
  { key: SERVED, name: "Trees, Platt scaling" },
];

function interval(value: { ci: [number | null, number | null] } | undefined): [number | null, number | null] | null {
  return value ? value.ci : null;
}

function calibrationOption(evaluation: SilenceEvaluation): ChartOption {
  const lines = [
    { key: "glm", name: "Logistic GLM", colour: "token:chart-neutral" },
    { key: SERVED, name: "Served score", colour: "token:accent" },
  ].filter((line) => evaluation.reliability[line.key]?.length);
  return {
    tooltip: {
      trigger: "item",
      formatter: (item: unknown) => {
        const { seriesName, value, color } = item as { seriesName: string; value: [number, number, number]; color: string };
        return tooltipTitle(seriesName) + tooltipRow(color, `${formatShare(value[1])} observed`, `at ${formatShare(value[0])} predicted, n = ${formatCount(value[2])}`);
      },
    },
    legend: { top: 0, left: 0, data: lines.map((line) => line.name) },
    xAxis: { type: "value", min: 0, max: 1, name: "predicted", nameLocation: "middle", nameGap: 28, axisLabel: { formatter: (v: number) => formatShare(v, 0) } },
    yAxis: { type: "value", min: 0, max: 1, name: "observed", axisLabel: { formatter: (v: number) => formatShare(v, 0) } },
    series: [
      {
        type: "line",
        id: "perfect",
        name: "Perfect calibration",
        data: [
          [0, 0],
          [1, 1],
        ],
        symbol: "none",
        silent: true,
        lineStyle: { color: "token:chart-axis", width: 1 },
        tooltip: { show: false },
      },
      ...lines.map((line) => ({
        type: "line" as const,
        id: line.key,
        name: line.name,
        data: (evaluation.reliability[line.key] ?? []).map((bin) => [bin.predicted, bin.observed, bin.n]),
        itemStyle: { color: line.colour },
        lineStyle: { color: line.colour },
        z: line.key === SERVED ? 3 : 2,
      })),
    ],
  } as ChartOption;
}

function liftOption(evaluation: SilenceEvaluation): ChartOption {
  const models = MODELS.filter((model) => evaluation.metrics[model.key]?.lift_at_10);
  const lift = (key: string) => evaluation.metrics[key]?.lift_at_10;
  return {
    tooltip: {
      trigger: "item",
      formatter: (item: unknown) => {
        const { dataIndex, color } = item as { dataIndex: number; color: string };
        const model = models[dataIndex];
        const value = model ? lift(model.key) : undefined;
        if (!model || !value) return "";
        return tooltipTitle(model.name) + tooltipRow(color, formatFixed(value.value, value.ci, 2, "×"), "lift in the top tenth");
      },
    },
    grid: { left: 0, right: 16, top: 8, bottom: 24, containLabel: true },
    xAxis: { type: "value", min: 0, name: "lift in the top tenth", nameLocation: "middle", nameGap: 28, axisLabel: { formatter: (v: number) => `${v}×` } },
    yAxis: { type: "category", inverse: true, data: models.map((model) => model.name), axisLine: { show: false } },
    series: [
      referenceLine({ id: "no-lift", at: 1, categories: models.length }),
      intervalSeries({ id: "lift-interval", intervals: models.map((model) => {
        const ci = interval(lift(model.key));
        return ci && ci[0] !== null && ci[1] !== null ? [ci[0], ci[1]] : null;
      }) }),
      {
        type: "scatter",
        id: "lift",
        symbolSize: 10,
        z: 5,
        data: models.map((model) => ({
          value: lift(model.key)?.value ?? null,
          itemStyle: { color: model.key === SERVED ? "token:accent" : "token:chart-neutral", borderColor: "token:chart-surface", borderWidth: 2 },
        })),
      },
    ],
  } as ChartOption;
}

export function RiskPage() {
  const evaluation = useQuery({ queryKey: ["silence-evaluation"], queryFn: api.silenceEvaluation });
  const ranked = useQuery({ queryKey: ["silence", TOP], queryFn: () => api.silence(TOP) });
  const data = evaluation.data;
  const calibration = useMemo(() => (data ? calibrationOption(data) : null), [data]);
  const lift = useMemo(() => (data ? liftOption(data) : null), [data]);
  const rows = ranked.data?.clients ?? [];

  return (
    <>
      <PageHeader
        title="Silence risk"
        lead="Who is likely to have no event in the three months after a cutoff. Every model learns on one cutoff and is tested on a later one, whose outcomes it never saw."
      />
      <div className="flex flex-col gap-5">
        <Panel
          title="Highest risks"
          subtitle={
            ranked.data?.cutoff
              ? `The ${formatCount(rows.length)} highest of ${formatCount(ranked.data.n_scored)} clients scored at the cutoff of ${formatDate(ranked.data.cutoff.slice(0, 10))}, by the served score (trees with Platt scaling).`
              : "The clients the served score ranks highest."
          }
        >
          {ranked.error ? <Failure error={ranked.error} onRetry={() => void ranked.refetch()} /> : null}
          {ranked.isPending ? (
            <Loading what="the ranked list">
              {[0, 1, 2, 3, 4, 5].map((index) => (
                <Bone key={index} className="h-7 w-full" />
              ))}
            </Loading>
          ) : null}
          {ranked.data && rows.length === 0 ? (
            <EmptyState title="No client is scored yet." command="just train">
              The ranked list appears once the silence model is trained on this workspace.
            </EmptyState>
          ) : null}
          {rows.length ? (
            <ScrollArea label="Clients ranked by silence risk">
              <table className="data-table min-w-[40rem]">
                <thead>
                  <tr>
                    <th scope="col" className="num">
                      Rank
                    </th>
                    <th scope="col">Client</th>
                    <th scope="col">Probability of silence</th>
                    <th scope="col">What pushes it up the most</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, index) => (
                    <tr key={row.client_id}>
                      <td className="num text-ink-secondary">{formatCount(index + 1)}</td>
                      <td className="font-mono text-label">{row.client_id}</td>
                      <td>
                        <span className="flex items-center gap-3">
                          <span className="h-1.5 w-24 shrink-0 overflow-hidden rounded-full bg-fill-control" aria-hidden>
                            <span className="block h-full origin-left rounded-full bg-accent" style={{ transform: `scaleX(${row.probability})` }} />
                          </span>
                          <span className="font-label text-ink">{formatShare(row.probability)}</span>
                        </span>
                      </td>
                      <td className="text-ink-secondary">{row.reasons.map((reason) => featureLabel(String(reason.feature))).join(", ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ScrollArea>
          ) : null}
        </Panel>

        {evaluation.error ? <Failure error={evaluation.error} onRetry={() => void evaluation.refetch()} /> : null}
        {evaluation.isPending ? (
          <Loading what="the evaluation">
            <div className="grid gap-5 xl:grid-cols-2">
              <Bone className="h-72 w-full" />
              <Bone className="h-72 w-full" />
            </div>
          </Loading>
        ) : null}
        {data && calibration && lift ? (
          <>
            <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
              <Panel title="Calibration" subtitle="Observed against predicted silence, by bin of the score, on the test cutoff. The diagonal is perfect calibration.">
                <Chart option={calibration} label="Reliability diagram of the served score and the logistic GLM" />
              </Panel>
              <Panel title="Lift in the top tenth" subtitle={`How much richer in silent clients the top tenth of each model's score is than the whole test set, with the ${data.interval}. One means no lift.`}>
                <Chart option={lift} label="Dot plot of each model's lift in the top tenth, with intervals" />
              </Panel>
            </div>
            <Panel
              title="Models compared"
              subtitle={`Trained at the cutoff of ${formatDate(data.train_cutoff.slice(0, 10))} on ${formatCount(data.n_train)} clients, tested at ${formatDate(data.test_cutoff.slice(0, 10))} on ${formatCount(data.n_test)}, whose silence rate is ${formatShare(data.silence_rate_test)}. Every value carries its ${data.interval}.`}
            >
              <ScrollArea label="Model comparison">
                <table className="data-table min-w-[44rem]">
                  <thead>
                    <tr>
                      <th scope="col">Model</th>
                      <th scope="col" className="num">
                        ROC AUC
                      </th>
                      <th scope="col" className="num">
                        Brier score
                      </th>
                      <th scope="col" className="num">
                        Lift in the top tenth
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {MODELS.filter((model) => data.metrics[model.key]).map((model) => {
                      const metrics = data.metrics[model.key];
                      return (
                        <tr key={model.key} className={model.key === SERVED ? "bg-accent-tint" : undefined}>
                          <td>
                            <span className="flex flex-wrap items-center gap-2">
                              <span className={model.key === SERVED ? "font-label text-ink" : "text-ink"}>{model.name}</span>
                              {model.key === SERVED ? <Tag>served</Tag> : null}
                            </span>
                          </td>
                          <td className="num">{formatFixed(metrics?.auc?.value, interval(metrics?.auc), 3)}</td>
                          <td className="num">{formatFixed(metrics?.brier?.value, interval(metrics?.brier), 3)}</td>
                          <td className="num">{formatFixed(metrics?.lift_at_10?.value, interval(metrics?.lift_at_10), 2, "×")}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </ScrollArea>
            </Panel>
          </>
        ) : null}
      </div>
    </>
  );
}
