import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

// Page titles say what each page holds; Findings and Alerts count what they show.
const PAGES: [string, RegExp][] = [
  ["/", /^Overview$/],
  ["/findings", /^Findings \(\d+ claims\)$/],
  ["/client", /^Customer profile$/],
  ["/segments", /^Segments$/],
  ["/risk", /^Silence risk$/],
  ["/adoption", /^Adoption$/],
  ["/alerts", /^Alerts \(\d+ ranked\)$/],
  ["/simulator", /^Simulator$/],
  ["/methods", /^Methods$/],
];

async function settled(page: Page) {
  // Loading states gone, no failure shown.
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
  test(`${title.source}: renders from the API, accessible in both themes, no sideways scroll`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(`/#${path}`);
    await expect(page.getByRole("heading", { level: 1, name: title })).toBeVisible();
    await settled(page);
    // The top bar names the data the page shows.
    await expect(page.getByRole("banner").getByText("Synthetic demo")).toBeVisible();
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
  await expect(page.getByText(/^\S+ \(\S+ to \S+\)$/).first()).toBeVisible();
  const verdictOnly = cards.filter({ hasText: "Published as a verdict only" });
  await expect(verdictOnly).toHaveCount(1);
  expect(await verdictOnly.innerText()).not.toMatch(/[0-9]/);
});

test("Simulator: the sliders drive POST /markov/simulate, the policy on screen first", async ({ page }) => {
  const first = page.waitForRequest((request) => request.url().endsWith("/api/markov/simulate") && request.method() === "POST");
  await page.goto("/#/simulator");
  expect(JSON.parse((await first).postData() ?? "{}")).toMatchObject({ share: 0.3, max_age: 45, target: "mixed" });
  await settled(page);

  const moved = page.waitForRequest(
    (request) => request.url().endsWith("/api/markov/simulate") && JSON.parse(request.postData() ?? "{}").share === 0.35,
  );
  await page.getByRole("slider", { name: "Share of eligible clients who move" }).focus();
  await page.keyboard.press("ArrowRight");
  const request = await moved;
  expect(JSON.parse(request.postData() ?? "{}")).toMatchObject({ max_age: 45, target: "mixed" });
  expect((await request.response())?.status()).toBe(200);
  await expect(page.getByText("Move 35% of branch-only clients under 45 to mixed use")).toBeVisible();
  await expect(page.getByRole("img", { name: /Digital share over the coming months/ })).toBeVisible();
});

test("Simulator: dragging the thumb moves the value with the pointer and settles on a step", async ({ page }) => {
  await page.goto("/#/simulator");
  await settled(page);
  const slider = page.getByRole("slider", { name: "Share of eligible clients who move" });
  const track = slider.locator("..");
  const box = await track.boundingBox();
  const thumb = await slider.boundingBox();
  if (!box || !thumb) throw new Error("the slider has no box");
  const y = thumb.y + thumb.height / 2;
  await page.mouse.move(thumb.x + thumb.width / 2, y);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.8, y, { steps: 8 });
  await expect(slider).toHaveAttribute("aria-valuenow", "0.8");
  // Past the end the thumb resists, and the value stays at the end.
  await page.mouse.move(box.x + box.width + 200, y, { steps: 8 });
  await expect(slider).toHaveAttribute("aria-valuenow", "1");
  await page.mouse.up();
  await expect(slider).toHaveAttribute("aria-valuenow", "1");
  await expect(page.getByText("Move 100% of branch-only clients under 45 to mixed use")).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("Alerts: evidence opens in a drawer, feedback there is stored and confirmed in place", async ({ page }) => {
  await page.goto("/#/alerts");
  await settled(page);
  await page.getByRole("button", { name: "Evidence for alert 1", exact: true }).click();
  const drawer = page.getByRole("dialog", { name: "Alert 1" });
  await expect(drawer).toBeVisible();
  const posted = page.waitForResponse((response) => /\/api\/alerts\/\d+\/feedback$/.test(response.url()));
  await drawer.getByRole("group", { name: "Feedback on alert 1", exact: true }).getByRole("button", { name: "Needs information" }).click();
  expect((await posted).status()).toBe(201);
  await expect(drawer.getByText(/The review is stored with the alert/)).toBeVisible();
  expect(await seriousViolations(page), "drawer open").toEqual([]);
  await page.keyboard.press("Escape");
  await expect(drawer).toBeHidden();
  await expect(page.getByRole("button", { name: "Evidence for alert 1", exact: true })).toBeFocused();
});
