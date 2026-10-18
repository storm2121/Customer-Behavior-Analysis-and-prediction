// Hand-made responses for component tests and stories. The values are illustrative round numbers,
// not results: real results only ever arrive from the API at run time.
import type {
  Adoption,
  Alert,
  Claims,
  Client360,
  Health,
  Overview,
  Segments,
  SilenceEvaluation,
  SilenceRisk,
  Simulation,
} from "./api";

export const health: Health = {
  status: "ok",
  zone: "public",
  versions: { strata: "0.1.0", python: "3.12" },
  data: { source_kind: "synthetic", fingerprint: "fixture-fingerprint", mapping: null },
  models: { silence: "0.1.0" },
};

export const overview: Overview = {
  source_kind: "synthetic",
  clients: 1000,
  events: 50000,
  channel_mix: [
    { channel_profile: "branch_only", clients: 400 },
    { channel_profile: "mixed", clients: 500 },
    { channel_profile: "none", clients: 100 },
  ],
  daily: [
    { day: "2024-01-01T00:00:00.000", channel: "card", events: 100 },
    { day: "2024-01-02T00:00:00.000", channel: "branch", events: 120 },
  ],
};

export const claims: Claims = {
  data_fingerprint: "fixture-fingerprint",
  claims: [
    {
      id: "digital_lift",
      title: "Digital clients keep more activity",
      question: "Do clients who use digital channels stay active longer?",
      values: { share_digital: 0.6, share_branch: 0.4, difference: 0.2, n_digital: 300 },
      units: { share_digital: "share", share_branch: "share", difference: "difference", n_digital: "count" },
      ci: { share_digital: [0.55, 0.65], share_branch: [0.35, 0.45], difference: [0.1, 0.3] },
      interval: "95% client bootstrap",
      n: 1000,
      verdict: "holds",
    },
  ],
  verdicts: [
    {
      id: "silence_by_channel",
      title: "Silence by channel",
      question: "Are branch-only clients more often silent than digital ones?",
      publication: "verdict_only",
      verdict: "reverses",
      stratifiers: ["activity_tercile", "age_band"],
      reversing_stratifiers: ["activity_tercile"],
      mechanism: "Branch-only clients are less active to begin with; within activity levels the gap turns around.",
      data_fingerprint: "fixture-fingerprint",
      source_kind: "synthetic",
    },
  ],
};

export const silence: SilenceRisk = {
  cutoff: "2024-09-30",
  clients: [
    {
      client_id: "fixture_a",
      cutoff_date: "2024-09-30",
      probability: 0.8,
      reasons: [{ feature: "recency_days", shap: 0.5 }],
    },
    {
      client_id: "fixture_b",
      cutoff_date: "2024-09-30",
      probability: 0.6,
      reasons: [{ feature: "active_months", shap: 0.3 }],
    },
  ],
};

const metric = (value: number) => ({ value, ci: [value - 0.05, value + 0.05] as [number, number] });

export const evaluation: SilenceEvaluation = {
  train_cutoff: "2024-06-30",
  test_cutoff: "2024-09-30",
  n_train: 800,
  n_test: 800,
  silence_rate_test: 0.2,
  metrics: {
    glm: { auc: metric(0.7), brier: metric(0.15), lift_at_10: metric(2) },
    lightgbm_platt: { auc: metric(0.75), brier: metric(0.14), lift_at_10: metric(2.5) },
  },
  reliability: {
    glm: [{ bin_low: 0, bin_high: 0.5, n: 400, predicted: 0.2, observed: 0.25 }],
    lightgbm_platt: [{ bin_low: 0, bin_high: 0.5, n: 400, predicted: 0.2, observed: 0.2 }],
  },
};

export const segments: Segments = {
  k: 2,
  stability: { mean_ari: 0.9, ci_low: 0.85, ci_high: 0.95 },
  segments: [
    { segment: 0, name: "card first", n_clients: 600, share: 0.6, medians: {}, channel_profile: {} },
    { segment: 1, name: "branch first", n_clients: 400, share: 0.4, medians: {}, channel_profile: {} },
  ],
  tests: {},
};

export const adoption: Adoption = {
  target_has_online: {
    rate: 0.3,
    n_clients: 500,
    deciles: [
      { decile: 1, n: 50, rate: 0.6, lift: 2, cumulative_gain: 0.2 },
      { decile: 2, n: 50, rate: 0.3, lift: 1, cumulative_gain: 0.3 },
    ],
  },
};

export const alert: Alert = {
  id: 1,
  client_id: "fixture_a",
  rank: 1,
  score: 0.9,
  rules: ["structuring"],
  reasons: [
    { source: "rule", rule: "structuring", evidence: {} },
    { source: "benford", detail: "leading digits far from Benford's law" },
  ],
  status: "open",
  label: null,
};

export const client360: Client360 = {
  client_id: "fixture_a",
  profile: { age: 40, channel_profile: "mixed", n_events: 120 },
  timeline: [
    { month_start: "2024-01-01T00:00:00.000", channel_state: "mixed", n_events: 10 },
    { month_start: "2024-02-01T00:00:00.000", channel_state: "branch_only", n_events: 4 },
  ],
  segment: 0,
  silence_risk: silence.clients[0] ?? null,
  open_alerts: [alert],
};

const band = (value: number) => ({ p05: [value - 0.1, value - 0.1], p50: [value, value], p95: [value + 0.1, value + 0.1] });

export const simulation: Simulation = {
  policy: { share: 0.3, max_age: 45, target: "mixed", horizon: 12, samples: 600 },
  eligible_bands: ["<25", "25-34", "35-44"],
  months: [0, 1],
  policy_bands: { digital_share: band(0.6), branch_load: band(500), silent_share: band(0.1) },
  baseline_bands: { digital_share: band(0.5), branch_load: band(600), silent_share: band(0.1) },
};

/** A fetch that answers every endpoint with the fixtures above, and records the calls. */
export function fixtureFetch(): { fetch: typeof fetch; calls: { url: string; init?: RequestInit }[] } {
  const calls: { url: string; init?: RequestInit }[] = [];
  const routes: [RegExp, unknown][] = [
    [/\/health$/, health],
    [/\/overview$/, overview],
    [/\/ledger\/claims$/, claims],
    [/\/clients\/[^/]+\/360$/, client360],
    [/\/segments\/points/, { points: [{ client_id: "fixture_a", segment: 0, x: 0, y: 0 }] }],
    [/\/segments$/, segments],
    [/\/risk\/silence\/evaluation$/, evaluation],
    [/\/risk\/silence/, silence],
    [/\/adoption\/deciles$/, adoption],
    [/\/alerts\/\d+\/feedback$/, { feedback_id: 1, alert_id: 1, label: "dismissed" }],
    [/\/alerts/, { alerts: [alert] }],
    [/\/markov\/simulate$/, simulation],
  ];
  const fake = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    calls.push({ url, init });
    const match = routes.find(([pattern]) => pattern.test(url.split("?")[0] ?? url) || pattern.test(url));
    if (!match) return new Response(JSON.stringify({ detail: "not found" }), { status: 404 });
    return new Response(JSON.stringify(match[1]), { status: 200, headers: { "Content-Type": "application/json" } });
  };
  return { fetch: fake as typeof fetch, calls };
}
