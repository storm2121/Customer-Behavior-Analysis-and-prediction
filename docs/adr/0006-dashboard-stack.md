# ADR 0006: React 19 with TypeScript for the dashboard

Status: accepted (architect, 2026-10-18). Filled in M0; expanded in M8, when the dashboard was built.

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

### How it was built (M8)

- **Data only at run time.** Every number on every page comes from the API through TanStack Query;
  the public claims arrive from `GET /ledger/claims`, so the bundle holds no claims file at all.
  `scripts/check-bundle.mjs` runs after every production build and fails it when a built file holds
  a value or the fingerprint of the public ledger, a token shaped like an identifier (the privacy
  guard's fallback pattern), a synthetic client id, or any file that is not code, styles, fonts or
  icons. Findings name the file and the rule, never the token.
- **One origin.** In Compose, nginx serves the build and proxies `/api/` to the API, so the browser
  needs no CORS and no API address is baked in; in development Vite proxies `/api` the same way
  (`STRATA_API_URL` moves the target). The API keeps a CORS allowlist (`STRATA_CORS_ORIGINS`) for
  other set-ups. A hash router keeps every page a static file, so any static host can serve it.
- **Verdict-only claims stay words.** The Findings page renders allow-listed claims with their
  values, intervals, sample sizes and robustness badges, and verdict-only claims with the verdict,
  the stratifier names and the mechanism sentence; a unit test and the smoke test assert that the
  verdict-only card contains no digit.
- **Design tokens.** Colours, radius and fonts are CSS variables with a light and a dark set (the
  system preference, or a choice remembered per browser); Tailwind utilities and the chart colours
  read the same tokens. Text tokens keep WCAG AA contrast on the surfaces and tints they sit on.
- **Weight.** ECharts is imported per chart type, every page is a lazy chunk, and KaTeX loads only
  with the Methods page, so the first paint does not wait for every chart and equation.
- **Tests.** Vitest and Testing Library cover the formatting, the claim cards, the theme switch, the
  feedback call and the simulator (each slider change posts the new policy to `/markov/simulate`),
  with charts stubbed because jsdom has no canvas. Storybook documents the components with
  illustrative fixtures and the accessibility panel. Playwright smoke tests run against a running
  stack on desktop and phone viewports: each page renders from the API with no failure, has no
  serious or critical axe violation in either theme and no sideways scroll; the simulator posts its
  sliders; feedback on an alert is stored. Locally they can drive an installed browser
  (`STRATA_E2E_CHANNEL`) instead of downloading one.

### The design pass (docs/DESIGN.md)

The architect's design brief extends this record; where the dashboard departs from the brief, and
why, is in `docs/DESIGN_NOTES.md`.

- **Tokens.** `src/styles/tokens.css` is the single source of every colour, type size, space, radius,
  material, elevation and duration, with light and dark values (the system preference, a chosen
  theme, or a stamped subtree, which is how Storybook shows both themes side by side) and the
  reduced-transparency and more-contrast preferences. Tailwind's own palette, radii, shadows and
  type scale are cleared, so a utility can only name a token. `src/styles/tokens.test.ts` keeps the
  computable promises: the dark blocks agree, every token the chart theme reads exists, text keeps
  WCAG AA on the surfaces it sits on, and the five categorical hues keep equal lightness and the
  colour-vision separation they were chosen for.
- **Charts.** The ECharts theme is generated from the tokens at run time on each chart's own element
  and merged into the option, so one chart instance lives as long as its page: data changes go
  through ECharts' transitions (series matched by id), the entrance plays only on a page's first
  paint, and reduced motion turns animation off. Options name colours by token (`token:<name>`).
- **Motion and icons.** Motion (`motion/react`) drives every spring: critically damped by default,
  a little bounce only after a drag release, a short cross-fade under reduced motion. Lucide is the
  only icon set. Both are new dependencies, locked at releases at least two weeks old.
- **Primitives.** The Radix slider is replaced by the dashboard's own slider, because the brief's
  showcase (one-to-one tracking under pointer capture, rubber-banding, the release velocity handed to
  a spring) is not something a stock slider does; it keeps the slider role, its values and the
  keyboard steps. Radix Dialog gives the evidence drawer its focus handling; the unused Radix tabs
  and tooltip are gone.
- **Context from the API.** The pages' numbers carry their interval, sample and period because the
  API serves them: the claims' method, caveats, robustness explanation and period, the overview's
  channel-mix intervals from a ledger of the same data, the segments' embedding, the scored
  population's size and the description of every interval it serves. All of it is additive.
- **Tests.** The smoke tests check the new page titles and data label, a pointer drag on the
  simulator's slider (held at the end, past which it resists), and feedback given in the drawer with
  axe clean while it is open. `scripts/screenshots.mjs` captures every page but Findings in both
  themes at a desktop and a phone width, after every chart reports it has finished drawing.

## Alternatives considered

- **Streamlit or Dash.** Fast to build, but design control, component testing and accessibility are
  limited.
- **Vue 3.** Already showcased by the sibling project.
- **Server-rendered templates.** Simple, but the simulator and the review queue are interactive.
- **Bundling the public claims JSON.** Allowed by the architecture, but fetching it keeps one
  source of truth and lets the bundle check be absolute: no ledger value in the bundle at all.
- **The full ECharts build.** Simpler imports, but a much larger bundle for charts the pages never
  draw.
- **Registering the ECharts theme (`registerTheme`).** The usual route, but ECharts 5 binds a theme
  when a chart is created, so a theme switch would remount every chart and replay its entrance.
- **Keeping the Radix slider.** Accessible and small, but it cannot rubber-band, hand off velocity
  or report its value while it springs back; the simulator's slider needs all three.

## Consequences

More build tooling and a longer M8, in exchange for a real product surface and a documented component
library. The dashboard never bundles private numbers, nor any data: it reads the public claims and every
other figure from the API, and its build fails if that ever changes.
