import { render, screen, within } from "@testing-library/react";

import { claims } from "../test-fixtures";
import { FindingCard, VerdictCard } from "./Findings";

const claim = claims.claims[0]!;
const verdict = claims.verdicts[0]!;

describe("FindingCard", () => {
  it("shows each value with its interval, the clients behind it and the robustness badge", () => {
    render(<FindingCard claim={claim} />);
    expect(screen.getByRole("heading", { name: claim.title })).toBeInTheDocument();
    expect(screen.getByText("Robustness: holds")).toBeInTheDocument();
    expect(screen.getByText("share digital")).toBeInTheDocument();
    expect(screen.getByText("(95% client bootstrap 55.0% to 65.0%)")).toBeInTheDocument();
    expect(screen.getByText("+20.0 points")).toBeInTheDocument();
    expect(screen.getByText("1,000 clients")).toBeInTheDocument();
  });

  it("lists the values that carry an interval first", () => {
    render(<FindingCard claim={claim} />);
    const terms = screen.getAllByRole("term").map((term) => term.textContent);
    expect(terms).toEqual(["share digital", "share branch", "difference", "n digital"]);
  });

  it("shows no badge for a claim the checker has not judged", () => {
    render(<FindingCard claim={{ ...claim, verdict: null }} />);
    expect(screen.queryByText(/Robustness:/)).not.toBeInTheDocument();
  });
});

describe("VerdictCard", () => {
  it("publishes a verdict-only claim in words, with no number at all", () => {
    const { container } = render(<VerdictCard verdict={verdict} />);
    const card = within(container);
    expect(card.getByText("Robustness: reverses")).toBeInTheDocument();
    expect(card.getByText("Checked within: activity level, age band")).toBeInTheDocument();
    expect(card.getByText("Reverses within: activity level")).toBeInTheDocument();
    expect(card.getByText(verdict.mechanism)).toBeInTheDocument();
    expect(container.textContent).not.toMatch(/[0-9]/);
  });
});
