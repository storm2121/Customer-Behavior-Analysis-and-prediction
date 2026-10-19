import type { PublicClaim, PublishedVerdict } from "../api";
import { formatCount, formatEstimate, formatPeriod, humanize, label } from "../format";
import { VerdictBadge } from "./ui";

const SHOWN = 8;

const SOURCE: Record<string, string> = {
  real: "reference extract, allow-listed aggregate",
  synthetic: "synthetic demo data",
};

/**
 * One public claim: the question it answers as its title, its values with their intervals in one
 * format, the robustness verdict with the sentence that explains it (or, without a verdict, how
 * the claim was measured), and the sample and period in a caption.
 */
export function FindingCard({ claim }: { claim: PublicClaim }) {
  // Values with an interval first: they are the claim's estimates; the rest are context.
  const all = Object.keys(claim.values);
  const keys = [...all.filter((key) => claim.ci[key]), ...all.filter((key) => !claim.ci[key])].slice(0, SHOWN);
  const hidden = all.length - keys.length;
  const caption = [
    `n = ${formatCount(claim.n)} clients`,
    claim.n_events === null ? null : `${formatCount(claim.n_events)} events`,
    formatPeriod(claim.period),
    `intervals are the ${claim.interval}`,
    SOURCE[claim.source_kind] ?? claim.source_kind,
  ].filter(Boolean);
  return (
    <article className="flex min-w-0 flex-col gap-4 rounded-panel border border-border bg-surface p-5">
      <header className="flex flex-col gap-1.5">
        <p className="text-label font-label text-ink-secondary">{claim.title}</p>
        <h2 className="text-heading font-title text-ink">{claim.question}</h2>
      </header>
      {claim.verdict ? (
        <div className="flex flex-col gap-2">
          <VerdictBadge verdict={claim.verdict} />
          {claim.mechanism ? <p className="max-w-measure text-small text-ink">{claim.mechanism}</p> : null}
        </div>
      ) : (
        <p className="max-w-measure text-small text-ink-secondary">{claim.method}</p>
      )}
      <dl className="grid grid-cols-1 gap-x-6 gap-y-2.5 sm:grid-cols-2">
        {keys.map((key) => (
          <div key={key} className="flex min-w-0 flex-col">
            <dt className="truncate text-label text-ink-secondary">{label(key)}</dt>
            <dd className="text-small font-label text-ink">
              {formatEstimate(claim.values[key] ?? Number.NaN, claim.ci[key], claim.units[key])}
            </dd>
          </div>
        ))}
      </dl>
      {hidden > 0 || claim.caveats.length || claim.verdict ? (
        <details className="group text-small text-ink-secondary">
          <summary className="flex min-h-11 cursor-pointer select-none items-center gap-1 font-label text-ink sm:min-h-8">
            {[claim.verdict ? "Method" : null, claim.caveats.length ? "Caveats" : null, hidden > 0 ? `${formatCount(hidden)} more values` : null]
              .filter(Boolean)
              .join(", ")}
          </summary>
          <div className="mt-2 flex max-w-measure flex-col gap-2">
            {claim.verdict ? <p>{claim.method}</p> : null}
            {claim.caveats.length ? (
              <ul className="flex list-disc flex-col gap-1 pl-5">
                {claim.caveats.map((caveat) => (
                  <li key={caveat}>{caveat}</li>
                ))}
              </ul>
            ) : null}
            {hidden > 0 ? <p>The other values of this claim are in the ledger and the findings documents.</p> : null}
          </div>
        </details>
      ) : null}
      <footer className="border-t border-border pt-3 text-label text-ink-secondary">{caption.join(" · ")}</footer>
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
    <article className="flex min-w-0 flex-col gap-4 rounded-panel border border-border bg-surface p-5">
      <header className="flex flex-col gap-1.5">
        <p className="text-label font-label text-ink-secondary">{verdict.title}</p>
        <h2 className="text-heading font-title text-ink">{verdict.question}</h2>
      </header>
      <div className="flex flex-col gap-2">
        <VerdictBadge verdict={verdict.verdict} />
        <p className="max-w-measure text-small text-ink">{verdict.mechanism}</p>
      </div>
      <dl className="flex flex-col gap-2 text-small">
        <div className="flex flex-col">
          <dt className="text-label text-ink-secondary">Checked within</dt>
          <dd className="font-label text-ink">{verdict.stratifiers.map(words).join(", ")}</dd>
        </div>
        {verdict.reversing_stratifiers.length ? (
          <div className="flex flex-col">
            <dt className="text-label text-ink-secondary">Reverses within</dt>
            <dd className="font-label text-ink">{verdict.reversing_stratifiers.map(words).join(", ")}</dd>
          </div>
        ) : null}
      </dl>
      <footer className="border-t border-border pt-3 text-label text-ink-secondary">
        Published as a verdict only: its numbers stay private.
      </footer>
    </article>
  );
}
