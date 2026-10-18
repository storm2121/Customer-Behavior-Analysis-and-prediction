import { defineConfig, devices } from "@playwright/test";

// Smoke tests against a running stack (`docker compose up`): every page renders from the API, with
// no serious accessibility violation in either theme and no horizontal scroll at phone width.
// STRATA_DASHBOARD_URL points at the dashboard (default: the Compose port on this machine);
// STRATA_E2E_CHANNEL=msedge (or chrome) drives an installed browser instead of a downloaded one.
const baseURL = process.env.STRATA_DASHBOARD_URL ?? "http://127.0.0.1:5173";
const channel = process.env.STRATA_E2E_CHANNEL || undefined;

export default defineConfig({
  testDir: "e2e",
  timeout: 90_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  use: { baseURL, channel, trace: "retain-on-failure" },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], channel } },
    { name: "phone", use: { ...devices["Pixel 7"], channel } },
  ],
});
