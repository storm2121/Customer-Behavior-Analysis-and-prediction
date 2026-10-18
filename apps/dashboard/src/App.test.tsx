import { QueryClient } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App, PAGES } from "./App";
import { fixtureFetch } from "./test-fixtures";

vi.mock("./components/Chart", () => ({
  Chart: ({ label }: { label: string }) => <div role="img" aria-label={label} />,
}));

const TITLES: Record<string, string> = {
  "/": "Overview",
  "/findings": "Findings",
  "/client": "Customer 360",
  "/segments": "Segments",
  "/risk": "Silence risk",
  "/adoption": "Online adoption",
  "/alerts": "Alerts",
  "/simulator": "What-if simulator",
  "/methods": "Methods",
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

  it("has nine pages", () => {
    expect(PAGES.map((page) => page.path)).toEqual(Object.keys(TITLES));
  });

  it.each(Object.entries(TITLES))("renders the page at %s from the API", async (path, title) => {
    renderAt(path);
    expect(await screen.findByRole("heading", { level: 1, name: title })).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole("status")).not.toBeInTheDocument());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shows the findings with intervals and badges, and the verdict-only claim without numbers", async () => {
    renderAt("/findings");
    const heading = await screen.findByRole("heading", { name: "Digital clients keep more activity" });
    const finding = heading.closest("article") as HTMLElement;
    expect(within(finding).getByText("Robustness: holds")).toBeInTheDocument();
    expect(within(finding).getByText(/55\.0% to 65\.0%/)).toBeInTheDocument();
    const verdict = screen.getByRole("heading", { name: "Silence by channel" }).closest("article") as HTMLElement;
    expect(verdict.textContent).not.toMatch(/[0-9]/);
    expect(within(verdict).getByText("Robustness: reverses")).toBeInTheDocument();
  });

  it("sends feedback on an alert", async () => {
    const user = userEvent.setup();
    renderAt("/alerts");
    const group = await screen.findByRole("group", { name: "Feedback on alert 1" });
    await user.click(within(group).getByRole("button", { name: "Dismiss" }));
    await waitFor(() => {
      const posted = fixture.calls.find((call) => call.url.endsWith("/alerts/1/feedback"));
      expect(posted?.init?.method).toBe("POST");
      expect(JSON.parse(String(posted?.init?.body))).toEqual({ label: "dismissed", analyst: "dashboard" });
    });
  });

  it("switches between the system, light and dark themes", async () => {
    const user = userEvent.setup();
    renderAt("/methods");
    await user.click(screen.getByRole("button", { name: /System theme/ }));
    expect(document.documentElement.dataset.theme).toBe("light");
    await user.click(screen.getByRole("button", { name: /Light theme/ }));
    expect(document.documentElement.dataset.theme).toBe("dark");
    await user.click(screen.getByRole("button", { name: /Dark theme/ }));
    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
  });
});
