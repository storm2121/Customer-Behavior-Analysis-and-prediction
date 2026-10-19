import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { fixtureFetch } from "../test-fixtures";
import { SimulatorPage } from "./Simulator";

vi.mock("../components/Chart", () => ({
  Chart: ({ label }: { label: string }) => <div role="img" aria-label={label} />,
}));

type Call = { url: string; init?: RequestInit };

function simulations(calls: Call[]) {
  return calls.filter((call) => call.url.endsWith("/markov/simulate")).map((call) => ({ method: call.init?.method, body: JSON.parse(String(call.init?.body)) }));
}

describe("the simulator page", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("posts the policy on screen first, then each settled change of a slider or the target", async () => {
    const fixture = fixtureFetch();
    vi.stubGlobal("fetch", fixture.fetch);
    const user = userEvent.setup();
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <SimulatorPage />
      </QueryClientProvider>,
    );

    await waitFor(() => expect(simulations(fixture.calls).length).toBeGreaterThan(0));
    expect(simulations(fixture.calls)[0]).toEqual({
      method: "POST",
      body: { share: 0.3, max_age: 45, target: "mixed", horizon: 12, samples: 600 },
    });
    expect(await screen.findByRole("img", { name: /Digital share over the coming months/ })).toBeInTheDocument();
    expect(screen.getByText("Eligible age bands: <25, 25-34, 35-44")).toBeInTheDocument();
    expect(screen.getByText("Move 30% of branch-only clients under 45 to mixed use")).toBeInTheDocument();

    // At rest the bands are the API's own answer: nothing is marked as updating.
    await waitFor(() => expect(screen.queryByText(/^Updating/)).not.toBeInTheDocument());
    const share = screen.getByRole("slider", { name: "Share of eligible clients who move" });
    share.focus();
    await user.keyboard("{ArrowRight}");
    expect(share).toHaveAttribute("aria-valuenow", "0.35");
    // Until the API answers for the new value, the interpolated bands are marked as provisional.
    expect(screen.getByText(/^Updating: these bands are interpolated/)).toBeInTheDocument();
    await waitFor(() => expect(simulations(fixture.calls).some((call) => call.body.share === 0.35 && call.body.max_age === 45)).toBe(true));
    await waitFor(() => expect(screen.queryByText(/^Updating/)).not.toBeInTheDocument());

    await user.keyboard("{Shift>}{ArrowRight}{/Shift}");
    expect(share).toHaveAttribute("aria-valuenow", "0.85");

    const age = screen.getByRole("slider", { name: "Clients younger than" });
    age.focus();
    await user.keyboard("{ArrowRight}");
    expect(age).toHaveAttribute("aria-valuetext", "55 years");
    await waitFor(() => expect(simulations(fixture.calls).some((call) => call.body.max_age === 55 && call.body.share === 0.85)).toBe(true));

    await user.click(screen.getByRole("radio", { name: "Digital only" }));
    await waitFor(() => expect(simulations(fixture.calls).some((call) => call.body.target === "digital_only")).toBe(true));
    expect(screen.getByText("Move 85% of branch-only clients under 55 to digital only")).toBeInTheDocument();
  });
});
