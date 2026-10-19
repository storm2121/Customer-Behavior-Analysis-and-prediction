import { render, screen, within } from "@testing-library/react";

import { claims } from "../test-fixtures";
import { FindingCard, VerdictCard } from "./Findings";

const claim = claims.claims[0]!;
const verdict = claims.verdicts[0]!;

describe("FindingCard", () => {
  it("titles the card with the question and reads each value with its interval in one format", () => {
    render(<FindingCard claim={claim} />);
    expect(screen.getByRole("heading", { name: claim.question })).toBeInTheDocument();
    expect(screen.getByText(claim.title)).toBeInTheDocument();
    expect(screen.getByText("Robustness: holds")).toBeInTheDocument();
    expect(screen.getByText(claim.mechanism!)).toBeInTheDocument();
    expect(screen.getByText("Share digital")).toBeInTheDocument();
    expect(screen.getByText("60.0% (55.0 to 65.0)")).toBeInTheDocument();
    expect(screen.getByText("+20.0 points (+10.0 to +30.0)")).toBeInTheDocument();
  });

  it("carries the sample, the period, the kind of interval and the source in its caption", () => {
    const { container } = render(<FindingCard claim={claim} />);
    const caption = container.querySelector("footer") as HTMLElement;
    expect(caption.textContent).toContain("n = 1,000 clients");
    expect(caption.textContent).toContain("50,000 events");
    expect(caption.textContent).toContain("2 October 2023 to 2 October 2024");
    expect(caption.textContent).toContain("95% client bootstrap");
    expect(caption.textContent).toContain("synthetic demo data");
  });

  it("lists the values that carry an interval first", () => {
    render(<FindingCard claim={claim} />);
    const terms = screen.getAllByRole("term").map((term) => term.textContent);
    expect(terms).toEqual(["Share digital", "Share branch", "Difference", "N digital"]);
  });

  it("shows the method in place of the verdict for a claim the checker has not judged", () => {
    render(<FindingCard claim={{ ...claim, verdict: null, mechanism: null }} />);
    expect(screen.queryByText(/Robustness:/)).not.toBeInTheDocument();
    expect(screen.getByText(claim.method)).toBeInTheDocument();
  });
});

describe("VerdictCard", () => {
  it("publishes a verdict-only claim in words, with no number at all", () => {
    const { container } = render(<VerdictCard verdict={verdict} />);
    const card = within(container);
    expect(card.getByRole("heading", { name: verdict.question })).toBeInTheDocument();
    expect(card.getByText("Robustness: reverses")).toBeInTheDocument();
    expect(card.getByText("activity level, age band")).toBeInTheDocument();
    expect(card.getByText("activity level")).toBeInTheDocument();
    expect(card.getByText(verdict.mechanism)).toBeInTheDocument();
    expect(card.getByText("Published as a verdict only: its numbers stay private.")).toBeInTheDocument();
    expect(container.textContent).not.toMatch(/[0-9]/);
  });
});
