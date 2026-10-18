import type { ReactNode } from "react";

export function Card({
  title,
  subtitle,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-card border border-border bg-surface p-4 shadow-sm sm:p-5 ${className}`}
      aria-label={title}
    >
      {title ? <h2 className="text-base font-semibold text-ink">{title}</h2> : null}
      {subtitle ? <p className="mt-1 text-sm text-ink-muted">{subtitle}</p> : null}
      <div className={title || subtitle ? "mt-3" : ""}>{children}</div>
    </section>
  );
}

export function Kpi({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-card border border-border bg-surface p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-ink-muted">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums text-ink">{value}</p>
      {hint ? <p className="mt-1 text-xs text-ink-muted">{hint}</p> : null}
    </div>
  );
}

const VERDICT_STYLES: Record<string, string> = {
  holds: "bg-positive/15 text-positive border-positive/40",
  weakens: "bg-warning/15 text-warning border-warning/40",
  reverses: "bg-negative/15 text-negative border-negative/40",
  insufficient: "bg-neutral/15 text-neutral border-neutral/40",
};

const VERDICT_WORDS: Record<string, string> = {
  holds: "Robustness: holds",
  weakens: "Robustness: weakens",
  reverses: "Robustness: reverses",
  insufficient: "Robustness: insufficient data",
};

export function VerdictBadge({ verdict }: { verdict: string }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold ${
        VERDICT_STYLES[verdict] ?? VERDICT_STYLES.insufficient
      }`}
    >
      {VERDICT_WORDS[verdict] ?? `Robustness: ${verdict}`}
    </span>
  );
}

export function Tag({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-md bg-surface-muted px-2 py-0.5 text-xs text-ink-muted">
      {children}
    </span>
  );
}

export function Loading({ what }: { what: string }) {
  return (
    <p role="status" className="text-sm text-ink-muted">
      Loading {what}…
    </p>
  );
}

export function Failure({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : "something went wrong";
  return (
    <p role="alert" className="rounded-card border border-negative/40 bg-negative/10 p-3 text-sm text-negative">
      The API did not answer: {message}. Is it running (`docker compose up`)?
    </p>
  );
}

export function PageHeader({ title, lead }: { title: string; lead: string }) {
  return (
    <header className="mb-5">
      <h1 className="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">{title}</h1>
      <p className="mt-2 max-w-3xl text-sm text-ink-muted sm:text-base">{lead}</p>
    </header>
  );
}

/** A horizontally scrollable region (wide tables on a phone), reachable from the keyboard. */
export function ScrollArea({ label, children }: { label: string; children: ReactNode }) {
  return (
    // A scrollable region needs a tab stop so keyboard users can scroll it.
    <div role="region" aria-label={label} tabIndex={0} className="overflow-x-auto rounded-md">
      {children}
    </div>
  );
}
