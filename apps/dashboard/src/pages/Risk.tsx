import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { api, type MetricValue } from "../api";
import { Chart, type ChartOption } from "../components/Chart";
import { Card, Failure, Loading, PageHeader, ScrollArea } from "../components/ui";
import { formatNumber, formatShare, humanize } from "../format";

const MODELS = ["majority", "activity_rule", "glm", "lightgbm", "lightgbm_isotonic", "lightgbm_platt"];

function metric(value: MetricValue | undefined, digits = 3): string {
  if (!value || value.value === null) return "not defined";
  const [low, high] = value.ci;
  const interval = low === null || high === null ? "" : ` (${formatNumber(low, digits)} to ${formatNumber(high, digits)})`;
  return `${formatNumber(value.value, digits)}${interval}`;
}

export function RiskPage() {
  const evaluation = useQuery({ queryKey: ["silence-evaluation"], queryFn: api.silenceEvaluation });
  const ranked = useQuery({ queryKey: ["silence", 25], queryFn: () => api.silence(25) });

  const calibration = useMemo<ChartOption>(() => {
    const bins = evaluation.data?.reliability ?? {};
    const lines = ["glm", "lightgbm_platt"].filter((name) => bins[name]);
    return {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      xAxis: { type: "value", name: "predicted", min: 0, max: 1 },
      yAxis: { type: "value", name: "observed", min: 0, max: 1 },
      series: [
        { type: "line", name: "perfect calibration", data: [[0, 0], [1, 1]], symbol: "none", lineStyle: { type: "dashed" } },
        ...lines.map((name) => ({
          type: "line" as const,
          name: humanize(name),
          data: (bins[name] ?? []).map((bin) => [bin.predicted, bin.observed]),
        })),
      ],
    };
  }, [evaluation.data]);

  const lift = useMemo<ChartOption>(() => {
    const metrics = evaluation.data?.metrics ?? {};
    const names = MODELS.filter((name) => metrics[name]);
    return {
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: names.map(humanize), axisLabel: { interval: 0, rotate: 20 } },
      yAxis: { type: "value", name: "lift at 10%" },
      series: [{ type: "bar", data: names.map((name) => metrics[name]?.lift_at_10?.value ?? 0), barMaxWidth: 40 }],
    };
  }, [evaluation.data]);

  return (
    <>
      <PageHeader
        title="Silence risk"
        lead="Who is likely to have no event in the three months after a cutoff. Every model learns on one cutoff and is tested on a later one; intervals are client bootstraps."
      />
      {evaluation.isPending ? <Loading what="the evaluation" /> : null}
      {evaluation.error ? <Failure error={evaluation.error} /> : null}
      {evaluation.data ? (
        <div className="flex flex-col gap-4">
          <Card
            title="Models compared"
            subtitle={`Trained on the cutoff of ${evaluation.data.train_cutoff}, tested on ${evaluation.data.test_cutoff}; test silence rate ${formatShare(evaluation.data.silence_rate_test)}.`}
          >
            <ScrollArea label="Model comparison">
              <table className="w-full min-w-[40rem] text-left text-sm">
                <thead className="text-xs uppercase tracking-wide text-ink-muted">
                  <tr>
                    <th className="py-2 pr-3">Model</th>
                    <th className="py-2 pr-3">ROC AUC</th>
                    <th className="py-2 pr-3">Brier</th>
                    <th className="py-2">Lift at 10%</th>
                  </tr>
                </thead>
                <tbody>
                  {MODELS.filter((name) => evaluation.data.metrics[name]).map((name) => (
                    <tr key={name} className="border-t border-border">
                      <td className="py-2 pr-3">{humanize(name)}</td>
                      <td className="py-2 pr-3 tabular-nums">{metric(evaluation.data.metrics[name]?.auc)}</td>
                      <td className="py-2 pr-3 tabular-nums">{metric(evaluation.data.metrics[name]?.brier)}</td>
                      <td className="py-2 tabular-nums">{metric(evaluation.data.metrics[name]?.lift_at_10, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ScrollArea>
          </Card>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <Card title="Calibration" subtitle="Observed against predicted silence, by bin of the score.">
              <Chart option={calibration} label="Reliability diagram of the calibrated models" height={300} />
            </Card>
            <Card title="Lift at ten per cent" subtitle="How much richer in silent clients the top tenth is than average.">
              <Chart option={lift} label="Bar chart of lift at ten per cent by model" height={300} />
            </Card>
          </div>
        </div>
      ) : null}
      <div className="mt-4">
        <Card title="Highest risks" subtitle="The served score: trees with Platt scaling.">
          {ranked.error ? <Failure error={ranked.error} /> : null}
          <ol className="flex flex-col gap-2">
            {(ranked.data?.clients ?? []).map((row, index) => (
              <li key={row.client_id} className="flex flex-wrap items-center gap-3 text-sm">
                <span className="w-6 text-right tabular-nums text-ink-muted">{index + 1}</span>
                <span className="font-mono text-xs">{row.client_id}</span>
                <span className="h-2 w-32 overflow-hidden rounded-full bg-surface-muted" aria-hidden>
                  <span className="block h-full bg-accent" style={{ width: `${Math.round(100 * row.probability)}%` }} />
                </span>
                <span className="tabular-nums">{formatShare(row.probability)}</span>
                <span className="text-ink-muted">
                  {row.reasons.slice(0, 2).map((reason) => humanize(String(reason.feature))).join(", ")}
                </span>
              </li>
            ))}
          </ol>
        </Card>
      </div>
    </>
  );
}
