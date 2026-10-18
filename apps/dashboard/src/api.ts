// The API's responses, typed, and the functions that fetch them. Nothing here holds data: every
// number the dashboard shows is fetched at run time from the API (synthetic data in Compose).

const BASE = (import.meta.env.VITE_API_URL ?? "/api").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // keep the status text
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export type Interval = [number, number];

export interface Health {
  status: string;
  zone: string;
  versions: { strata: string; python: string };
  data: { source_kind: string; fingerprint: string; mapping: string | null };
  models: Record<string, string>;
}

export interface Overview {
  source_kind: string;
  clients: number;
  events: number;
  channel_mix: { channel_profile: string; clients: number }[];
  daily: { day: string; channel: string; events: number }[];
}

export interface PublicClaim {
  id: string;
  title: string;
  question?: string;
  values: Record<string, number>;
  units?: Record<string, string>;
  ci: Record<string, Interval>;
  interval: string;
  n: number;
  verdict: string | null;
}

export interface PublishedVerdict {
  id: string;
  title: string;
  question: string;
  publication: "verdict_only";
  verdict: string;
  stratifiers: string[];
  reversing_stratifiers: string[];
  mechanism: string;
  data_fingerprint: string;
  source_kind: string;
}

export interface Claims {
  data_fingerprint: string;
  claims: PublicClaim[];
  verdicts: PublishedVerdict[];
}

export interface Reason {
  feature?: string;
  shap?: number;
  source?: string;
  rule?: string;
  detail?: string;
  evidence?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ScoredClient {
  client_id: string;
  cutoff_date: string;
  probability: number;
  reasons: Reason[];
}

export interface SilenceRisk {
  cutoff: string | null;
  clients: ScoredClient[];
}

export interface MetricValue {
  value: number | null;
  ci: [number | null, number | null];
}

export interface ReliabilityBin {
  bin_low: number;
  bin_high: number;
  n: number;
  predicted: number;
  observed: number;
}

export interface SilenceEvaluation {
  train_cutoff: string;
  test_cutoff: string;
  n_train: number;
  n_test: number;
  silence_rate_test: number;
  metrics: Record<string, Record<string, MetricValue>>;
  reliability: Record<string, ReliabilityBin[]>;
}

export interface Segment {
  segment: number;
  name: string;
  n_clients: number;
  share: number;
  medians: Record<string, number | null>;
  channel_profile: Record<string, number>;
}

export interface Segments {
  k: number;
  stability: { mean_ari: number; ci_low: number; ci_high: number };
  segments: Segment[];
  tests: Record<string, Record<string, number>>;
}

export interface SegmentPoint {
  client_id: string;
  segment: number;
  x: number;
  y: number;
}

export interface Decile {
  decile: number;
  n: number;
  rate: number;
  lift: number;
  cumulative_gain: number;
}

export type Adoption = Record<string, { rate: number; n_clients: number; deciles: Decile[] }>;

export interface Alert {
  id: number;
  client_id: string;
  rank: number;
  score: number;
  rules: string[];
  reasons: Reason[];
  status: string;
  label: string | null;
}

export interface Client360 {
  client_id: string;
  profile: Record<string, unknown>;
  timeline: { month_start: string; channel_state: string; n_events: number }[];
  segment: number | null;
  silence_risk: ScoredClient | null;
  open_alerts: Alert[];
}

export interface CalendarEffects {
  [claim: string]: {
    interval: string;
    effects: Record<string, { value: number; ci: Interval | null }>;
    caveats: string[];
  };
}

export interface SimulationRequest {
  share: number;
  max_age: number;
  target: "mixed" | "digital_only" | "none";
  horizon: number;
  samples: number;
}

export type Bands = Record<string, { p05: number[]; p50: number[]; p95: number[] }>;

export interface Simulation {
  policy: SimulationRequest;
  eligible_bands: string[];
  months: number[];
  policy_bands: Bands;
  baseline_bands: Bands;
}

export const api = {
  health: () => request<Health>("/health"),
  overview: () => request<Overview>("/overview"),
  claims: () => request<Claims>("/ledger/claims"),
  client: (id: string) => request<Client360>(`/clients/${encodeURIComponent(id)}/360`),
  segments: () => request<Segments>("/segments"),
  segmentPoints: () => request<{ points: SegmentPoint[] }>("/segments/points?limit=5000"),
  silence: (top: number) => request<SilenceRisk>(`/risk/silence?top=${top}`),
  silenceEvaluation: () => request<SilenceEvaluation>("/risk/silence/evaluation"),
  adoption: () => request<Adoption>("/adoption/deciles"),
  alerts: (budget: number) => request<{ alerts: Alert[] }>(`/alerts?budget=${budget}`),
  feedback: (id: number, label: string, analyst: string) =>
    request<{ feedback_id: number }>(`/alerts/${id}/feedback`, {
      method: "POST",
      body: JSON.stringify({ label, analyst }),
    }),
  calendar: () => request<CalendarEffects>("/calendar/effects"),
  simulate: (body: SimulationRequest) =>
    request<Simulation>("/markov/simulate", { method: "POST", body: JSON.stringify(body) }),
};
