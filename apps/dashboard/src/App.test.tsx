import { QueryClient } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App, PAGES } from "./App";
import { fixtureFetch } from "./test-fixtures";

vi.mock("./components/Chart", () => ({
  Chart: ({ label }: { label: string }) => <div role="img" aria-label={label} />,
}));

// Page titles say what the page holds; two of them count what they show.
const TITLES: Record<string, RegExp> = {
  "/": /^Overview$/,
  "/findings": /^Findings \(2 claims\)$/,
  "/client": /^Customer profile$/,
  "/segments": /^Segments$/,
  "/risk": /^Silence risk$/,
  "/adoption": /^Adoption$/,
  "/alerts": /^Alerts \(1 ranked\)$/,
  "/simulator": /^Simulator$/,
  "/methods": /^Methods$/,
};

function renderAt(path: string) {
  window.location.hash = `#${path}`;
  return render(<App client={new QueryClient({ defaultOptions: { queries: { retry: false } } })} />);
}

describe("the dashboard", () => {
  let fixture: ReturnType<typeof fixtureFetch>;
  beforeEach(() => {
    fixture = fixtureFetch();
    vi.stubGlobal("fetch", fixture.fetch);
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("has nine pages, none called Home or Dashboard", () => {
    expect(PAGES.map((page) => page.path)).toEqual(Object.keys(TITLES));
    for (const page of PAGES) expect(page.label).not.toMatch(/home|dashboard/i);
  });

  it.each(Object.entries(TITLES))("renders the page at %s from the API", async (path, title) => {
    renderAt(path);
    // Each page is a lazy chunk; Methods brings KaTeX, which takes a moment to import.
    expect(await screen.findByRole("heading", { level: 1, name: title }, { timeout: 10_000 })).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole("status")).not.toBeInTheDocument(), { timeout: 10_000 });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("names the page and the data it shows in the top bar", async () => {
    renderAt("/segments");
    const banner = screen.getByRole("banner");
    expect(banner.querySelector("p")?.textContent).toContain("Segments");
    expect(within(banner).getByRole("link", { name: "Segments", current: "page" })).toBeInTheDocument();
    expect(await within(banner).findByText("Synthetic demo")).toBeInTheDocument();
  });

  it("shows the findings with intervals and badges, and the verdict-only claim without numbers", async () => {
    renderAt("/findings");
    const heading = await screen.findByRole("heading", { name: "Do clients who use digital channels stay active longer?" });
    const finding = heading.closest("article") as HTMLElement;
    expect(within(finding).getByText("Robustness: holds")).toBeInTheDocument();
    expect(within(finding).getByText("60.0% (55.0 to 65.0)")).toBeInTheDocument();
    const verdict = screen.getByRole("heading", { name: "Are branch-only clients more often silent than digital ones?" }).closest("article") as HTMLElement;
    expect(verdict.textContent).not.toMatch(/[0-9]/);
    expect(within(verdict).getByText("Robustness: reverses")).toBeInTheDocument();
  });

  it("opens an alert's evidence in a drawer and stores the feedback given there", async () => {
    const user = userEvent.setup();
    renderAt("/alerts");
    await user.click(await screen.findByRole("button", { name: "Evidence for alert 1" }));
    const drawer = await screen.findByRole("dialog", { name: "Alert 1" });
    expect(within(drawer).getByText("Structuring")).toBeInTheDocument();
    const group = within(drawer).getByRole("group", { name: "Feedback on alert 1" });
    await user.click(within(group).getByRole("button", { name: "Dismiss" }));
    await waitFor(() => {
      const posted = fixture.calls.find((call) => call.url.endsWith("/alerts/1/feedback"));
      expect(posted?.init?.method).toBe("POST");
      expect(JSON.parse(String(posted?.init?.body))).toEqual({ label: "dismissed", analyst: "dashboard" });
    });
    expect(await within(drawer).findByText(/Dismissed\. The review is stored/)).toBeInTheDocument();
    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("switches between the system, light and dark themes", async () => {
    const user = userEvent.setup();
    renderAt("/methods");
    await user.click(await screen.findByRole("button", { name: /System theme/ }));
    expect(document.documentElement.dataset.theme).toBe("light");
    await user.click(screen.getByRole("button", { name: /Light theme/ }));
    expect(document.documentElement.dataset.theme).toBe("dark");
    await user.click(screen.getByRole("button", { name: /Dark theme/ }));
    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
  });
});
