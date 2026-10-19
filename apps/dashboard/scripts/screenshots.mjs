// Regenerates the dashboard screenshots (docs/images/dashboard-<page>-<theme>-<width>.png) from a
// running stack on synthetic data: `node scripts/screenshots.mjs` with the stack up. Every page is
// captured in the light and the dark theme, at a desktop width and at a phone width.
//
// It refuses to capture anything unless the API reports the public zone and a synthetic source, so
// no real figure can end up in an image. The Findings page is not captured: its values are the
// allow-listed public claims of the real extract, which belong in the ledger and the documents
// rendered from it (where `strata ledger check` keeps them current), not in pictures.
//
// STRATA_DASHBOARD_URL points at the dashboard; STRATA_E2E_CHANNEL=msedge drives an installed
// browser; STRATA_SCREENSHOT_DIR writes elsewhere (a review copy outside the repository).
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "@playwright/test";

const here = fileURLToPath(new URL(".", import.meta.url));
const out = process.env.STRATA_SCREENSHOT_DIR ?? resolve(here, "..", "..", "..", "docs", "images");
const base = (process.env.STRATA_DASHBOARD_URL ?? "http://127.0.0.1:5173").replace(/\/$/, "");
const channel = process.env.STRATA_E2E_CHANNEL || undefined;

const health = await (await fetch(`${base}/api/health`)).json();
if (health.zone !== "public" || health.data?.source_kind !== "synthetic") {
  console.error("screenshots: the API does not serve the public synthetic workspace; nothing captured");
  process.exit(1);
}
mkdirSync(out, { recursive: true });

const PAGES = [
  { name: "overview", path: "/", title: /^Overview$/ },
  { name: "client", path: "/client", title: /^Customer profile$/ },
  { name: "segments", path: "/segments", title: /^Segments$/ },
  { name: "risk", path: "/risk", title: /^Silence risk$/ },
  { name: "adoption", path: "/adoption", title: /^Adoption$/ },
  { name: "alerts", path: "/alerts", title: /^Alerts/ },
  { name: "simulator", path: "/simulator", title: /^Simulator$/ },
  { name: "methods", path: "/methods", title: /^Methods$/ },
];
const THEMES = ["light", "dark"];
const SCREENS = [
  { width: 1440, viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 },
  { width: 390, viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true },
];

/** Waits until the page holds no loading state and every chart has finished drawing. */
async function settled(page) {
  await page.waitForFunction(() => document.querySelectorAll("[role='status']").length === 0, null, { timeout: 30_000 });
  await page.waitForFunction(
    () => [...document.querySelectorAll("[data-chart]")].every((chart) => chart.getAttribute("data-settled") === "true"),
    null,
    { timeout: 30_000 },
  );
  await page.evaluate(() => document.fonts.ready);
  // Springs and fades have finished well within this; nothing on the page is still moving.
  await page.waitForTimeout(400);
}

const browser = await chromium.launch({ channel });
let count = 0;
try {
  for (const screen of SCREENS) {
    for (const theme of THEMES) {
      const { width: _width, ...options } = screen;
      const context = await browser.newContext({ ...options, colorScheme: theme });
      for (const target of PAGES) {
        const page = await context.newPage();
        await page.goto(`${base}/#${target.path}`);
        await page.getByRole("heading", { level: 1, name: target.title }).waitFor();
        await settled(page);
        const file = `dashboard-${target.name}-${theme}-${screen.width}.png`;
        await page.screenshot({ path: resolve(out, file) });
        await page.close();
        count += 1;
        console.log(`screenshots: ${file}`);
      }
      await context.close();
    }
  }
} finally {
  await browser.close();
}
console.log(`screenshots: ${count} captured`);
