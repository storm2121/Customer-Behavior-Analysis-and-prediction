import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

const PAGES: [string, string][] = [
  ["/", "Overview"],
  ["/findings", "Findings"],
  ["/client", "Customer 360"],
  ["/segments", "Segments"],
  ["/risk", "Silence risk"],
  ["/adoption", "Online adoption"],
  ["/alerts", "Alerts"],
  ["/simulator", "What-if simulator"],
  ["/methods", "Methods"],
];

async function settled(page: Page) {
  // Loading messages gone, no failure shown.
  await expect(page.getByText(/^Loading /)).toHaveCount(0);
  await expect(page.getByRole("alert")).toHaveCount(0);
}

async function seriousViolations(page: Page): Promise<string[]> {
  const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
  return results.violations
    .filter((violation) => violation.impact === "serious" || violation.impact === "critical")
    .map((violation) => `${violation.id} (${violation.nodes.length})`);
}

for (const [path, title] of PAGES) {
  test(`${title}: renders from the API, accessible in both themes, no sideways scroll`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(`/#${path}`);
    await expect(page.getByRole("heading", { level: 1, name: title })).toBeVisible();
    await settled(page);
    for (const theme of ["light", "dark"]) {
      await page.evaluate((value) => document.documentElement.setAttribute("data-theme", value), theme);
      expect(await seriousViolations(page), `${theme} theme`).toEqual([]);
    }
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(1);
    expect(errors).toEqual([]);
  });
}

test("Findings: public claims with intervals and badges, the verdict-only claim without numbers", async ({ page }) => {
  await page.goto("/#/findings");
  await settled(page);
  const cards = page.locator("article");
  await expect(cards.first()).toBeVisible();
  await expect(page.getByText(/^Robustness: /).first()).toBeVisible();
  await expect(page.getByText(/\bto\b/).first()).toBeVisible();
  const verdictOnly = cards.filter({ hasText: "Published as a verdict only" });
  await expect(verdictOnly).toHaveCount(1);
  expect(await verdictOnly.innerText()).not.toMatch(/[0-9]/);
});

test("Simulator: the sliders drive POST /markov/simulate", async ({ page }) => {
  const first = page.waitForRequest((request) => request.url().endsWith("/api/markov/simulate") && request.method() === "POST");
  await page.goto("/#/simulator");
  expect(JSON.parse((await first).postData() ?? "{}")).toMatchObject({ share: 0.3, max_age: 45, target: "mixed" });
  await settled(page);

  const moved = page.waitForRequest(
    (request) => request.url().endsWith("/api/markov/simulate") && JSON.parse(request.postData() ?? "{}").share > 0.3,
  );
  await page.getByRole("slider", { name: "Share of eligible clients who move" }).focus();
  await page.keyboard.press("ArrowRight");
  const request = await moved;
  expect(JSON.parse(request.postData() ?? "{}")).toMatchObject({ max_age: 45, target: "mixed" });
  expect((await request.response())?.status()).toBe(200);
  await expect(page.getByRole("img", { name: /Digital share over twelve months/ })).toBeVisible();
});

test("Alerts: feedback on an alert is stored", async ({ page }) => {
  await page.goto("/#/alerts");
  await settled(page);
  const posted = page.waitForResponse((response) => /\/api\/alerts\/\d+\/feedback$/.test(response.url()));
  await page.getByRole("group", { name: "Feedback on alert 1", exact: true }).getByRole("button", { name: "Needs information" }).click();
  expect((await posted).status()).toBe(201);
});
