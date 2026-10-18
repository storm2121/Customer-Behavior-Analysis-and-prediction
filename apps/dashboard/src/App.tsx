import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { lazy, Suspense, useEffect, useState, type ComponentType } from "react";
import { HashRouter, NavLink, Route, Routes } from "react-router";

import { Loading } from "./components/ui";
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

export const PAGES = [
  { path: "/", label: "Overview", element: <OverviewPage /> },
  { path: "/findings", label: "Findings", element: <FindingsPage /> },
  { path: "/client", label: "Customer 360", element: <ClientPage /> },
  { path: "/segments", label: "Segments", element: <SegmentsPage /> },
  { path: "/risk", label: "Risk", element: <RiskPage /> },
  { path: "/adoption", label: "Adoption", element: <AdoptionPage /> },
  { path: "/alerts", label: "Alerts", element: <AlertsPage /> },
  { path: "/simulator", label: "Simulator", element: <SimulatorPage /> },
  { path: "/methods", label: "Methods", element: <MethodsPage /> },
] as const;

const THEME_WORDS: Record<Theme, string> = { system: "System theme", light: "Light theme", dark: "Dark theme" };

function ThemeButton() {
  const [theme, setTheme] = useState<Theme>(storedTheme);
  useEffect(() => applyTheme(theme), [theme]);
  return (
    <button
      type="button"
      onClick={() => setTheme(nextTheme(theme))}
      className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm text-ink hover:bg-surface-muted"
      aria-label={`${THEME_WORDS[theme]}; switch theme`}
    >
      {THEME_WORDS[theme]}
    </button>
  );
}

export function Shell() {
  return (
    <div className="min-h-screen bg-canvas text-ink">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-10 focus:rounded-md focus:bg-surface focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <header className="sticky top-0 z-10 border-b border-border bg-surface/95 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3">
          <p className="text-lg font-semibold tracking-tight">
            Strata <span className="text-sm font-normal text-ink-muted">synthetic demo</span>
          </p>
          <ThemeButton />
        </div>
        <nav aria-label="Pages" className="mx-auto max-w-6xl overflow-x-auto px-2">
          <ul className="flex gap-1 whitespace-nowrap pb-2">
            {PAGES.map((item) => (
              <li key={item.path}>
                <NavLink
                  to={item.path}
                  end={item.path === "/"}
                  className={({ isActive }) =>
                    `block rounded-md px-3 py-1.5 text-sm ${
                      isActive ? "bg-accent text-accent-ink" : "text-ink-muted hover:bg-surface-muted hover:text-ink"
                    }`
                  }
                >
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </header>
      <main id="main" className="mx-auto max-w-6xl px-4 py-6">
        <Suspense fallback={<Loading what="the page" />}>
          <Routes>
            {PAGES.map((item) => (
              <Route key={item.path} path={item.path} element={item.element} />
            ))}
          </Routes>
        </Suspense>
      </main>
      <footer className="mx-auto max-w-6xl px-4 pb-8 text-xs text-ink-muted">
        Every number on these pages is fetched from the Strata API; the stack serves synthetic data,
        and the only real figures are the allow-listed public claims on the Findings page.
      </footer>
    </div>
  );
}

export function App({ client }: { client?: QueryClient }) {
  const [queryClient] = useState(
    () => client ?? new QueryClient({ defaultOptions: { queries: { staleTime: 60_000, retry: 1 } } }),
  );
  return (
    <QueryClientProvider client={queryClient}>
      <HashRouter>
        <Shell />
      </HashRouter>
    </QueryClientProvider>
  );
}
