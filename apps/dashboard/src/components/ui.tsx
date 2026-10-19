import {
  ArrowDownUp,
  Check,
  CircleAlert,
  CircleCheck,
  CircleDashed,
  Copy,
  Info,
  RotateCw,
  TrendingDown,
  TriangleAlert,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useId, useRef, useState, type ReactNode } from "react";

import { ApiError } from "../api";
import { Icon } from "./Icon";
import { Button } from "./Pressable";

/** The page title (what the page holds) and one plain sentence on what it shows. */
export function PageHeader({ title, lead, children }: { title: string; lead: string; children?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-col gap-2 sm:mb-8">
      <h1 className="text-title font-title text-ink">{title}</h1>
      <p className="max-w-measure text-body text-ink-secondary">{lead}</p>
      {children ? <div className="mt-1 flex flex-wrap gap-2">{children}</div> : null}
    </div>
  );
}

/** An in-flow panel: a hairline border on the surface, no shadow (shadows are for floating things). */
export function Panel({
  title,
  subtitle,
  actions,
  id,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
  id?: string;
  children: ReactNode;
  className?: string;
}) {
  const headingId = useId();
  return (
    <section
      id={id}
      aria-labelledby={title ? headingId : undefined}
      className={`min-w-0 scroll-mt-4 rounded-panel border border-border bg-surface p-4 sm:p-5 ${className}`}
    >
      {title || actions ? (
        <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div className="flex min-w-0 flex-col gap-1">
            {title ? (
              <h2 id={headingId} tabIndex={-1} className="text-heading font-title text-ink outline-none">
                {title}
              </h2>
            ) : null}
            {subtitle ? <p className="max-w-measure text-small text-ink-secondary">{subtitle}</p> : null}
          </div>
          {actions}
        </header>
      ) : null}
      {children}
    </section>
  );
}

/** A small label for a fact that travels with the content (the source, an eligibility). */
export function Tag({ children, icon }: { children: ReactNode; icon?: LucideIcon }) {
  return (
    <span className="inline-flex min-h-6 items-center gap-1.5 rounded-control bg-fill-control px-2 text-label font-label text-ink-secondary">
      {icon ? <Icon icon={icon} /> : null}
      {children}
    </span>
  );
}

/** A label, a number with its interval, and one line of context. Not a tile: a line of a column. */
export function Figure({
  label,
  value,
  interval,
  context,
}: {
  label: string;
  value: string;
  interval?: string;
  context: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <dt className="text-label font-label text-ink-secondary">{label}</dt>
      <dd className="flex flex-col gap-1">
        <span className="flex flex-wrap items-baseline gap-x-2">
          <span className="text-figure font-title text-ink">{value}</span>
          {interval ? <span className="text-small text-ink-secondary">{interval}</span> : null}
        </span>
        <span className="max-w-measure text-small text-ink-secondary">{context}</span>
      </dd>
    </div>
  );
}

type NoticeKind = "status" | "completion" | "warning" | "error";

const NOTICE: Record<NoticeKind, { icon: LucideIcon; tone: string }> = {
  status: { icon: Info, tone: "border-border bg-fill-subtle text-ink-secondary" },
  completion: { icon: CircleCheck, tone: "border-success/40 bg-success-tint text-success" },
  warning: { icon: TriangleAlert, tone: "border-warning/40 bg-warning-tint text-warning" },
  error: { icon: CircleAlert, tone: "border-danger/40 bg-danger-tint text-danger" },
};

/**
 * Feedback in one of four distinct kinds: status (what is going on), completion (it worked),
 * warning (something to know before going on) and error (what happened and what to do). Inline,
 * in place, never a toast. Errors are announced at once; the others politely.
 */
export function Notice({
  kind,
  title,
  children,
  action,
}: {
  kind: NoticeKind;
  title?: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  const { icon, tone } = NOTICE[kind];
  return (
    <div
      role={kind === "error" ? "alert" : undefined}
      className={`flex items-start gap-3 rounded-panel border px-4 py-3 text-small ${tone}`}
    >
      <span className="mt-0.5">
        <Icon icon={icon} />
      </span>
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        {title ? <p className="font-label text-ink">{title}</p> : null}
        {children ? <div className="max-w-measure text-ink">{children}</div> : null}
        {action ? <div className="mt-2 flex flex-wrap gap-2">{action}</div> : null}
      </div>
    </div>
  );
}

/** A command with a button that copies it, confirming in place. */
export function Command({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  const canCopy = typeof navigator !== "undefined" && Boolean(navigator.clipboard);
  return (
    <span className="inline-flex flex-wrap items-center gap-2">
      <code className="rounded-control bg-fill-control px-2 py-1 text-ink">{command}</code>
      {canCopy ? (
        <Button
          tone="quiet"
          icon={copied ? Check : Copy}
          onClick={() => {
            void navigator.clipboard.writeText(command).then(() => {
              setCopied(true);
              window.clearTimeout(timer.current);
              timer.current = window.setTimeout(() => setCopied(false), 2000);
            });
          }}
        >
          {copied ? "Copied" : "Copy"}
        </Button>
      ) : null}
    </span>
  );
}

/** What a failed request means for the reader, and the one command that fixes it. */
export function explain(error: unknown): { what: string; command: string | null } {
  if (error instanceof ApiError) {
    const detail = error.message;
    if (error.status === 503 && detail.includes("models/"))
      return { what: `The models behind this page are not trained in this workspace (${detail}).`, command: "just train" };
    if (error.status === 503 && detail.includes("analysis/"))
      return { what: `The analysis behind this page has not run in this workspace (${detail}).`, command: "just analyse" };
    if (error.status === 503 && detail.includes("ledger"))
      return { what: `The ledger behind this page is not built (${detail}).`, command: "just ledger" };
    return { what: `The API answered ${error.status}: ${detail}.`, command: null };
  }
  return { what: "The dashboard could not reach the API.", command: "just serve" };
}

/** An error that says what happened and what to do, with a retry in place. */
export function Failure({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const { what, command } = explain(error);
  return (
    <Notice
      kind="error"
      title="This part of the page could not load"
      action={
        <>
          {command ? <Command command={command} /> : null}
          {onRetry ? (
            <Button tone="secondary" icon={RotateCw} onClick={onRetry}>
              Retry
            </Button>
          ) : null}
        </>
      }
    >
      {what} {command ? "Run the command below, then retry." : "Retry once the API is back."}
    </Notice>
  );
}

/** What will appear here, how to make it appear, and the one action that does it. */
export function EmptyState({ title, children, command }: { title: string; children: ReactNode; command?: string }) {
  return (
    <div className="flex flex-col items-start gap-2 rounded-panel border border-dashed border-border-strong px-4 py-6">
      <p className="text-small font-label text-ink">{title}</p>
      <p className="max-w-measure text-small text-ink-secondary">{children}</p>
      {command ? <Command command={command} /> : null}
    </div>
  );
}

/** A placeholder shaped like the content it stands for. */
export function Bone({ className }: { className: string }) {
  return <span aria-hidden className={`skeleton block ${className}`} />;
}

/** A loading state: announced once, drawn as the content's shape. */
export function Loading({ what, children }: { what: string; children?: ReactNode }) {
  return (
    <div role="status" aria-busy="true" className="flex flex-col gap-4">
      <span className="sr-only">Loading {what}…</span>
      {children ?? (
        <div className="flex flex-col gap-3 rounded-panel border border-border bg-surface p-5">
          <Bone className="h-5 w-48" />
          <Bone className="h-4 w-80 max-w-full" />
          <Bone className="h-56 w-full" />
        </div>
      )}
    </div>
  );
}

/** The skeleton of a page while its code loads. */
export function PageSkeleton() {
  return (
    <Loading what="the page">
      <div className="mb-2 flex flex-col gap-3">
        <Bone className="h-8 w-56" />
        <Bone className="h-4 w-96 max-w-full" />
      </div>
      <div className="flex flex-col gap-3 rounded-panel border border-border bg-surface p-5">
        <Bone className="h-5 w-40" />
        <Bone className="h-64 w-full" />
      </div>
    </Loading>
  );
}

const VERDICTS: Record<string, { icon: LucideIcon; tone: string; words: string }> = {
  holds: { icon: CircleCheck, tone: "text-verdict-holds bg-verdict-holds/10 border-verdict-holds/35", words: "holds" },
  weakens: { icon: TrendingDown, tone: "text-verdict-weakens bg-verdict-weakens/10 border-verdict-weakens/35", words: "weakens" },
  reverses: { icon: ArrowDownUp, tone: "text-verdict-reverses bg-verdict-reverses/10 border-verdict-reverses/35", words: "reverses" },
  insufficient: { icon: CircleDashed, tone: "text-ink-secondary bg-fill-control border-border-strong", words: "insufficient data" },
};

/** A robustness verdict: its colour, its icon and its word, so no reader depends on colour. */
export function VerdictBadge({ verdict }: { verdict: string }) {
  const style = VERDICTS[verdict] ?? { ...VERDICTS.insufficient!, words: verdict };
  return (
    <span className={`inline-flex min-h-6 w-fit items-center gap-1.5 rounded-control border px-2 text-label font-label ${style.tone}`}>
      <Icon icon={style.icon} />
      <span>Robustness: {style.words}</span>
    </span>
  );
}

/**
 * A horizontally scrollable region (a wide table on a phone), reachable from the keyboard, with a
 * soft fade on the side that has more beyond it instead of a hard cut.
 */
export function ScrollArea({ label, children }: { label: string; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [more, setMore] = useState({ before: false, after: false });
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const measure = () => {
      const before = element.scrollLeft > 1;
      const after = element.scrollLeft + element.clientWidth < element.scrollWidth - 1;
      setMore((current) => (current.before === before && current.after === after ? current : { before, after }));
    };
    measure();
    element.addEventListener("scroll", measure, { passive: true });
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => {
      element.removeEventListener("scroll", measure);
      observer.disconnect();
    };
  }, []);
  return (
    <div
      ref={ref}
      role="region"
      aria-label={label}
      tabIndex={0}
      data-more-before={more.before ? "" : undefined}
      data-more-after={more.after ? "" : undefined}
      className="scroll-fade-x overflow-x-auto rounded-control"
    >
      {children}
    </div>
  );
}
