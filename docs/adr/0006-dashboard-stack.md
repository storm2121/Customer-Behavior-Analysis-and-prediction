# ADR 0006: React 19 with TypeScript for the dashboard

Status: accepted (architect, 2026-10-18). Filled in M0; expanded when the dashboard is built in M8.

## Context

The legacy UI was a Tkinter form. The rebuild needs a designed, accessible, testable interface with
rich charts, and the sibling project (DetectionX5) deliberately uses a different front-end stack
(Vue 3) so the two projects showcase different skills.

The dashboard must never bundle private numbers: it reads the public claims file and an API that
serves synthetic data.

## Decision

React 19, TypeScript strict, Vite, TanStack Query for server state, ECharts for charts, Tailwind with
design tokens and Radix primitives, KaTeX for equations, pnpm, Vitest, Storybook, Playwright smoke
tests, axe accessibility checks. Streamlit was rejected because design is a stated goal.

## Alternatives considered

- **Streamlit or Dash.** Fast to build, but design control, component testing and accessibility are
  limited.
- **Vue 3.** Already showcased by the sibling project.
- **Server-rendered templates.** Simple, but the simulator and the review queue are interactive.

## Consequences

More build tooling and a longer M8, in exchange for a real product surface and a documented component
library. The dashboard never bundles private numbers; it reads the public claims JSON and the API.
