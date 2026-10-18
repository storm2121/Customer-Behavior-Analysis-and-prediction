import type { PublicClaim, PublishedVerdict } from "../api";
import { formatCount, formatInterval, formatValue, humanize } from "../format";
import { Tag, VerdictBadge } from "./ui";

const SHOWN = 8;

/** One public claim: its values with their intervals, the clients behind it, its verdict. */
export function FindingCard({ claim }: { claim: PublicClaim }) {
  // Values with an interval first: they are the claim's estimates; the rest are context.
  const all = Object.keys(claim.values);
  const keys = [...all.filter((key) => claim.ci[key]), ...all.filter((key) => !claim.ci[key])].slice(0, SHOWN);
  return (
    <article className="flex flex-col gap-3 rounded-card border border-border bg-surface p-4 sm:p-5">
      <header className="flex flex-wrap items-start justify-between gap-2">
        <h2 className="text-base font-semibold text-ink">{claim.title}</h2>
        {claim.verdict ? <VerdictBadge verdict={claim.verdict} /> : null}
      </header>
      {claim.question ? <p className="text-sm text-ink-muted">{claim.question}</p> : null}
      <dl className="grid grid-cols-1 gap-x-4 gap-y-2 sm:grid-cols-2">
        {keys.map((key) => {
          const unit = claim.units?.[key];
          const interval = claim.ci[key];
          return (
            <div key={key} className="min-w-0">
              <dt className="truncate text-xs text-ink-muted">{humanize(key)}</dt>
              <dd className="text-sm font-medium tabular-nums text-ink">
                {formatValue(claim.values[key] ?? Number.NaN, unit)}
                {interval ? (
                  <span className="ml-1 text-xs font-normal text-ink-muted">
                    ({claim.interval} {formatInterval(interval, unit)})
                  </span>
                ) : null}
              </dd>
            </div>
          );
        })}
      </dl>
      <footer className="flex flex-wrap gap-2">
        <Tag>{formatCount(claim.n)} clients</Tag>
        {all.length > SHOWN ? (
          <Tag>{all.length - SHOWN} more values in the ledger</Tag>
        ) : null}
      </footer>
    </article>
  );
}

const STRATIFIER_WORDS: Record<string, string> = {
  activity_tercile: "activity level",
  age_band: "age band",
  tenure_band: "tenure band",
  gender: "gender code",
};

/** A headline published by its verdict only: words, never a number (by construction). */
export function VerdictCard({ verdict }: { verdict: PublishedVerdict }) {
  const words = (name: string) => STRATIFIER_WORDS[name] ?? humanize(name);
  return (
    <article className="flex flex-col gap-3 rounded-card border border-border bg-surface p-4 sm:p-5">
      <header className="flex flex-wrap items-start justify-between gap-2">
        <h2 className="text-base font-semibold text-ink">{verdict.title}</h2>
        <VerdictBadge verdict={verdict.verdict} />
      </header>
      <p className="text-sm text-ink-muted">{verdict.question}</p>
      <p className="text-sm text-ink">{verdict.mechanism}</p>
      <footer className="flex flex-wrap gap-2">
        <Tag>Published as a verdict only: its numbers stay private</Tag>
        <Tag>Checked within: {verdict.stratifiers.map(words).join(", ")}</Tag>
        {verdict.reversing_stratifiers.length ? (
          <Tag>Reverses within: {verdict.reversing_stratifiers.map(words).join(", ")}</Tag>
        ) : null}
      </footer>
    </article>
  );
}
