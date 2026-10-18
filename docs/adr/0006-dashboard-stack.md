# ADR 0006: React 19 with TypeScript for the dashboard

Status: accepted (architect, 2026-10-18). Expand during M8.

## Context

The legacy UI was a Tkinter form. The rebuild needs a designed, accessible, testable interface with
rich charts, and the sibling project (DetectionX5) deliberately uses a different front-end stack
(Vue 3) so the two projects showcase different skills.

## Decision

React 19, TypeScript strict, Vite, TanStack Query for server state, ECharts for charts, Tailwind with
design tokens and Radix primitives, KaTeX for equations, pnpm, Vitest, Storybook, Playwright smoke
tests, axe accessibility checks. Streamlit was rejected because design is a stated goal.

## Consequences

More build tooling and a longer M8, in exchange for a real product surface and a documented component
library. The dashboard never bundles private numbers; it reads the public claims JSON and the API.
