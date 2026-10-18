import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { fixtureFetch } from "../test-fixtures";
import { SimulatorPage } from "./Simulator";

vi.mock("../components/Chart", () => ({
  Chart: ({ label }: { label: string }) => <div role="img" aria-label={label} />,
}));

function lastSimulation(calls: { url: string; init?: RequestInit }[]) {
  const posted = calls.filter((call) => call.url.endsWith("/markov/simulate"));
  const last = posted[posted.length - 1];
  return last ? { method: last.init?.method, body: JSON.parse(String(last.init?.body)) } : null;
}

describe("the simulator page", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("posts the slider values to /markov/simulate and redraws on every change", async () => {
    const fixture = fixtureFetch();
    vi.stubGlobal("fetch", fixture.fetch);
    const user = userEvent.setup();
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <SimulatorPage />
      </QueryClientProvider>,
    );

    await waitFor(() => expect(lastSimulation(fixture.calls)).not.toBeNull());
    expect(lastSimulation(fixture.calls)).toEqual({
      method: "POST",
      body: { share: 0.3, max_age: 45, target: "mixed", horizon: 12, samples: 600 },
    });
    expect(await screen.findByRole("img", { name: /Digital share over twelve months/ })).toBeInTheDocument();
    expect(screen.getByText("Eligible age bands: <25, 25-34, 35-44")).toBeInTheDocument();

    screen.getByRole("slider", { name: "Share of eligible clients who move" }).focus();
    await user.keyboard("{ArrowRight}");
    await waitFor(() => expect(lastSimulation(fixture.calls)?.body.share).toBeCloseTo(0.35));

    screen.getByRole("slider", { name: "Clients younger than" }).focus();
    await user.keyboard("{ArrowRight}");
    await waitFor(() => expect(lastSimulation(fixture.calls)?.body.max_age).toBe(55));

    await user.selectOptions(screen.getByRole("combobox", { name: "They move to" }), "digital_only");
    await waitFor(() => expect(lastSimulation(fixture.calls)?.body.target).toBe("digital_only"));
  });
});
