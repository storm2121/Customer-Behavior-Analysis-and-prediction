// Regenerates the README screenshots (docs/images/dashboard-*.png) from a running stack on
// synthetic data: `node scripts/screenshots.mjs` with the stack up. It refuses to capture anything
// unless the API reports the public zone and a synthetic source, so no real figure can end up in an
// image. The Findings page is not captured: its values are the allow-listed public claims, which
// belong in the ledger and the documents rendered from it, not in pictures.
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "@playwright/test";

const here = fileURLToPath(new URL(".", import.meta.url));
const out = resolve(here, "..", "..", "..", "docs", "images");
const base = (process.env.STRATA_DASHBOARD_URL ?? "http://127.0.0.1:5173").replace(/\/$/, "");
const channel = process.env.STRATA_E2E_CHANNEL || undefined;

const health = await (await fetch(`${base}/api/health`)).json();
if (health.zone !== "public" || health.data?.source_kind !== "synthetic") {
  console.error("screenshots: the API does not serve the public synthetic workspace; nothing captured");
  process.exit(1);
}
mkdirSync(out, { recursive: true });

const SHOTS = [
  { name: "overview", path: "/", title: "Overview", scheme: "light" },
  { name: "risk", path: "/risk", title: "Silence risk", scheme: "light" },
  { name: "segments", path: "/segments", title: "Segments", scheme: "dark" },
  { name: "simulator", path: "/simulator", title: "What-if simulator", scheme: "light" },
  { name: "alerts", path: "/alerts", title: "Alerts", scheme: "dark" },
  { name: "phone-simulator", path: "/simulator", title: "What-if simulator", scheme: "dark", phone: true },
];

const browser = await chromium.launch({ channel });
try {
  for (const shot of SHOTS) {
    const page = await browser.newPage(
      shot.phone
        ? { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, colorScheme: shot.scheme }
        : { viewport: { width: 1280, height: 900 }, colorScheme: shot.scheme },
    );
    await page.goto(`${base}/#${shot.path}`);
    await page.getByRole("heading", { level: 1, name: shot.title }).waitFor();
    await page.getByText(/^Loading /).first().waitFor({ state: "detached" });
    await page.waitForTimeout(1500); // let the charts finish their entry animation
    await page.screenshot({ path: resolve(out, `dashboard-${shot.name}.png`) });
    await page.close();
    console.log(`screenshots: dashboard-${shot.name}.png`);
  }
} finally {
  await browser.close();
}
