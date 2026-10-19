import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { Monitor, Moon, Sun, type LucideIcon } from "lucide-react";
import { MotionConfig, motion } from "motion/react";
import { lazy, Suspense, useEffect, useRef, useState, type ComponentType } from "react";
import { HashRouter, NavLink, Route, Routes, useLocation } from "react-router";

import { api } from "./api";
import { Icon } from "./components/Icon";
import { PageSkeleton } from "./components/ui";
import { useScrollEdges } from "./hooks";
import { PRESS_SCALE, SPRING, useCalmMotion } from "./motion";
import { applyTheme, nextTheme, storedTheme, type Theme } from "./theme";

// Each page is its own chunk: the first paint does not wait for every chart and equation.
function page<K extends string>(load: () => Promise<Record<K, ComponentType>>, name: K) {
  return lazy(async () => ({ default: (await load())[name] }));
}

const OverviewPage = page(() => import("./pages/Overview"), "OverviewPage");
const FindingsPage = page(() => import("./pages/Findings"), "FindingsPage");
const ClientPage = page(() => import("./pages/Client"), "ClientPage");
const SegmentsPage = page(() => import("./pages/Segments"), "SegmentsPage");
const RiskPage = page(() => import("./pages/Risk"), "RiskPage");
const AdoptionPage = page(() => import("./pages/Adoption"), "AdoptionPage");
const AlertsPage = page(() => import("./pages/Alerts"), "AlertsPage");
const SimulatorPage = page(() => import("./pages/Simulator"), "SimulatorPage");
const MethodsPage = page(() => import("./pages/Methods"), "MethodsPage");

// Labels say what each page holds; there is no "Home" and no "Dashboard".
export const PAGES = [
  { path: "/", label: "Overview", element: <OverviewPage /> },
  { path: "/findings", label: "Findings", element: <FindingsPage /> },
  { path: "/client", label: "Customer profile", element: <ClientPage /> },
  { path: "/segments", label: "Segments", element: <SegmentsPage /> },
  { path: "/risk", label: "Silence risk", element: <RiskPage /> },
  { path: "/adoption", label: "Adoption", element: <AdoptionPage /> },
  { path: "/alerts", label: "Alerts", element: <AlertsPage /> },
  { path: "/simulator", label: "Simulator", element: <SimulatorPage /> },
  { path: "/methods", label: "Methods", element: <MethodsPage /> },
] as const;

const THEMES: Record<Theme, { words: string; short: string; icon: LucideIcon }> = {
  system: { words: "System theme", short: "System", icon: Monitor },
  light: { words: "Light theme", short: "Light", icon: Sun },
  dark: { words: "Dark theme", short: "Dark", icon: Moon },
};

const MotionLink = motion.create(NavLink);

function ThemeButton() {
  const [theme, setTheme] = useState<Theme>(storedTheme);
  const calm = useCalmMotion();
  useEffect(() => applyTheme(theme), [theme]);
  const { words, short, icon } = THEMES[theme];
  return (
    <motion.button
      type="button"
      onClick={() => setTheme(nextTheme(theme))}
      whileTap={calm ? undefined : { scale: PRESS_SCALE }}
      transition={SPRING}
      aria-label={`${words}; switch theme`}
      title="Switch between the system, light and dark themes"
      className="inline-flex min-h-11 min-w-11 select-none items-center justify-center gap-2 rounded-control px-2 text-small font-label text-ink-secondary hover:bg-fill-subtle hover:text-ink active:bg-fill-control sm:min-h-9 sm:px-3"
    >
      <Icon icon={icon} size={20} />
      <span className="hidden sm:inline">{short}</span>
    </motion.button>
  );
}

function NavItems({ layout }: { layout: "column" | "row" }) {
  const calm = useCalmMotion();
  return (
    <ul className={layout === "column" ? "flex flex-col gap-0.5" : "flex gap-1 whitespace-nowrap"}>
      {PAGES.map((item) => (
        <li key={item.path}>
          <MotionLink
            to={item.path}
            end={item.path === "/"}
            whileTap={calm ? undefined : { scale: PRESS_SCALE }}
            transition={SPRING}
            className={({ isActive }: { isActive: boolean }) =>
              `flex min-h-11 select-none items-center rounded-control px-3 text-small no-underline [-webkit-tap-highlight-color:transparent] lg:min-h-9 ${
                isActive
                  ? "bg-accent-tint font-label text-accent"
                  : "text-ink-secondary hover:bg-fill-subtle hover:text-ink active:bg-fill-control"
              }`
            }
          >
            {item.label}
          </MotionLink>
        </li>
      ))}
    </ul>
  );
}

/** The page list as a row that scrolls sideways, fading at the edge that has more beyond it. */
function NavRow() {
  const ref = useRef<HTMLDivElement | null>(null);
  const location = useLocation();
  const [measure, more] = useScrollEdges<HTMLDivElement>();
  const attach = (node: HTMLDivElement | null) => {
    ref.current = node;
    measure(node);
  };
  // Keep the current page's link in view when the route changes, scrolling the row only (never
  // the page).
  useEffect(() => {
    const row = ref.current;
    const link = row?.querySelector<HTMLElement>("[aria-current='page']");
    if (!row || !link) return;
    const box = row.getBoundingClientRect();
    const own = link.getBoundingClientRect();
    const start = own.left - box.left + row.scrollLeft;
    const end = start + own.width;
    if (start < row.scrollLeft) row.scrollLeft = start;
    else if (end > row.scrollLeft + row.clientWidth) row.scrollLeft = end - row.clientWidth;
  }, [location.pathname]);
  return (
    <nav aria-label="Pages" className="lg:hidden">
      <div
        ref={attach}
        data-more-before={more.before ? "" : undefined}
        data-more-after={more.after ? "" : undefined}
        className="scroll-fade-x overflow-x-auto px-[var(--gutter)] pb-1 [scrollbar-width:none]"
      >
        <NavItems layout="row" />
      </div>
    </nav>
  );
}

/** What the page shows: the synthetic demo everywhere, and on Findings the claims' own source. */
function DataLabel() {
  const location = useLocation();
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, staleTime: Number.POSITIVE_INFINITY });
  const onFindings = location.pathname === "/findings";
  const claims = useQuery({ queryKey: ["claims"], queryFn: api.claims, enabled: onFindings });
  const data = health.data;
  const stack =
    data === undefined
      ? null
      : data.zone === "public" && data.data.source_kind === "synthetic"
        ? "Synthetic demo"
        : data.zone === "private"
          ? "Private zone"
          : `${data.data.source_kind} data`;
  const claimsSource = onFindings && claims.data ? (claims.data.source_kind === "real" ? "Claims: reference extract" : "Claims: synthetic") : null;
  return (
    <span className="flex min-w-0 flex-wrap items-center justify-end gap-1.5">
      {stack ? <DataChip>{stack}</DataChip> : null}
      {claimsSource ? <DataChip>{claimsSource}</DataChip> : null}
    </span>
  );
}

function DataChip({ children }: { children: string }) {
  return (
    <span className="inline-flex min-h-6 items-center whitespace-nowrap rounded-control border border-border px-2 text-label font-label text-ink-secondary">
      {children}
    </span>
  );
}

function TopBar() {
  const location = useLocation();
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const update = () => setScrolled(window.scrollY > 0);
    update();
    window.addEventListener("scroll", update, { passive: true });
    return () => window.removeEventListener("scroll", update);
  }, []);
  const current = PAGES.find((item) => item.path === location.pathname)?.label ?? "Strata";
  return (
    <header data-scrolled={scrolled ? "" : undefined} className="material material-bar sticky top-0 z-30">
      <div className="flex min-h-14 items-center justify-between gap-3 px-[var(--gutter)]">
        <p className="flex min-w-0 items-baseline gap-2">
          <span className="hidden text-body font-title text-ink sm:inline lg:hidden">Strata</span>
          <span className="truncate text-small font-title text-ink">{current}</span>
        </p>
        <div className="flex min-w-0 items-center gap-2">
          <DataLabel />
          <ThemeButton />
        </div>
      </div>
      <NavRow />
      <div className="scroll-edge" aria-hidden />
    </header>
  );
}

function Footer() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, staleTime: Number.POSITIVE_INFINITY });
  return (
    <footer className="mt-auto max-w-measure px-[var(--gutter)] pt-4 pb-8 text-label text-ink-secondary">
      Every number on these pages is fetched from the Strata API. The stack serves synthetic data; the only real
      figures are the allow-listed public claims on the Findings page.
      {health.data ? ` Data fingerprint ${health.data.data.fingerprint.slice(0, 12)}.` : null}
    </footer>
  );
}

export function Shell() {
  return (
    <div className="min-h-dvh bg-canvas text-ink lg:grid lg:grid-cols-[var(--sidebar-width)_minmax(0,1fr)]">
      <a
        href="#main"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById("main")?.focus();
        }}
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded-control focus:bg-surface-raised focus:px-3 focus:py-2 focus:shadow-float"
      >
        Skip to content
      </a>
      <aside className="sticky top-0 hidden h-dvh flex-col gap-6 border-r border-border px-3 py-4 lg:flex">
        <p className="px-3 text-heading font-title text-ink">Strata</p>
        <nav aria-label="Pages">
          <NavItems layout="column" />
        </nav>
      </aside>
      <div className="flex min-h-dvh min-w-0 flex-col">
        <TopBar />
        <main id="main" tabIndex={-1} className="w-full max-w-7xl px-[var(--gutter)] pt-6 pb-10 outline-none sm:pt-8">
          <Suspense fallback={<PageSkeleton />}>
            <Routes>
              {PAGES.map((item) => (
                <Route key={item.path} path={item.path} element={item.element} />
              ))}
            </Routes>
          </Suspense>
        </main>
        <Footer />
      </div>
    </div>
  );
}

export function App({ client }: { client?: QueryClient }) {
  const [queryClient] = useState(
    () => client ?? new QueryClient({ defaultOptions: { queries: { staleTime: 60_000, retry: 1 } } }),
  );
  return (
    <QueryClientProvider client={queryClient}>
      <MotionConfig reducedMotion="user">
        <HashRouter>
          <Shell />
        </HashRouter>
      </MotionConfig>
    </QueryClientProvider>
  );
}
