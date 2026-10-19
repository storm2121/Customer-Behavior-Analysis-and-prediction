# Design brief: the Strata dashboard

Status: architect brief, 2026-10-19. The executing agent redesigns `apps/dashboard` against this
document. It extends `ARCHITECTURE.md` §13 and ADR 0006 and does not change what the dashboard does,
only how it looks, moves and reads. The principles are Apple's (WWDC design talks, translated to the
web): the interface should feel like an extension of the analyst using it, not like a template.

## 1. What it must not look like

The dashboard must not read as machine-generated. Remove, and never add:

- gradient hero blocks, purple-to-blue accents, glow effects, blurred colour blobs;
- rows of identical KPI cards with an icon, a title and a sentence each; a KPI needs a comparison
  or an interval or it is decoration;
- emoji as icons, mixed icon sets, decorative icons;
- rounded cards with drop shadows on everything; shadows are for floating surfaces only;
- generic copy: "Welcome", "Insights", "Powerful", "Seamless", "Overview of your data";
- rainbow categorical palettes, red-to-green scales, 3D or donut charts, dual y-axes;
- spinners where a skeleton or the content could appear; toasts for routine success;
- centred layouts; analytics reads left to right, top to bottom.

## 2. Foundations

- **Type.** The platform system stack: `-apple-system, BlinkMacSystemFont, "Segoe UI Variable",
  "Segoe UI", system-ui, sans-serif`, `font-optical-sizing: auto`, tabular figures everywhere a
  number appears. Size-specific tracking: titles at 28px and above `-0.02em`, 20px `-0.01em`, body
  `0`, 12px labels `+0.01em`. Leading 1.1 for titles, 1.45 for prose, 1.3 for dense tables. Weight
  carries hierarchy: 600 titles, 500 labels, 400 body. Sizes in rem. Equations in KaTeX at the body
  size with a 70-character measure.
- **Colour.** One accent (a restrained blue with AA contrast on both themes), one success, one
  warning, one danger, nine neutral greys, all tokens on `:root` and redefined for dark mode.
  Surfaces are near-white and near-black. Charts: the neutral ramp plus the accent for the thing
  being discussed; categorical series from a fixed set of five hues at equal lightness, documented
  in the tokens file; sequential scales single-hue; diverging scales only where zero means something,
  and then blue-to-orange, never red-to-green. Robustness verdicts have three fixed colours that
  are also distinguishable by their label and icon.
- **Space.** A 4px base scale. Page gutter 24px on desktop, 16px at phone width. Tables are dense
  (row height 36px) and text has a measure under 70 characters.
- **Shape.** Radius 6px for controls, 10px for panels, 16px for floating sheets. Hairline borders
  from the neutral ramp. No shadows on in-flow panels.
- **Materials.** The top bar and any sheet are translucent (`backdrop-filter: blur(20px)
  saturate(180%)` over a 60% surface with a bright top edge), content scrolls under them, never two
  translucent surfaces stacked, soft scroll-edge fades instead of divider lines. Respect
  `prefers-reduced-transparency` and `prefers-contrast: more`.
- **Icons.** Lucide only, 16px and 20px, stroke 1.75, always with a label or an accessible name.

## 3. Motion

- **Respond on pointer-down.** Controls press the instant they are touched. No animation locks input.
- **Springs for anything touchable.** The Motion library (`motion` for React) with critically damped
  springs by default (`bounce: 0`, `duration: 0.35`); bounce only after a drag release.
- **The simulator sliders** are the showcase: the thumb tracks the pointer 1:1 with pointer capture,
  the bands redraw continuously while dragging (debounce the network call, never the visual), the
  thumb rubber-bands at the ends, and release hands velocity to a spring. Keyboard arrows move by
  one step, Shift by ten.
- **Interruptible.** Every transition starts from the current on-screen value. Panels exit the way
  they entered; popovers scale from their trigger.
- **Charts** update through ECharts' own transitions on data change, never by remounting; entrance
  animation is off except for the first paint of a page.
- **Reduced motion**: every spring and slide becomes a 200ms cross-fade.
- **Only `transform` and `opacity` animate.**

## 4. Craft rules

- Every number carries its unit and its interval or sample size; the format is consistent within a
  table; percentages have one decimal at most; intervals read `48.4% (41.2 to 55.1)`.
- Claims render with their robustness badge and the sentence that explains the verdict; the
  verdict-only claim renders its words with no digits, as now.
- Feedback kinds are distinct: status, completion, warning, error. Errors say what happened and what
  to do. Alert feedback confirms inline, in place, without a toast.
- Empty states say what will appear and how to make it appear, with the one action that does it.
- Skeletons shaped like the content; tables paginate or virtualise beyond 200 rows.
- Keyboard reachable everywhere, visible focus rings, Escape closes sheets. Touch targets 44px at
  phone width.
- Wayfinding: the top bar names the page and the data it shows ("synthetic demo" stays visible on
  every page as a small label, since the public stack never shows real data). Page titles say what
  the page holds: "Findings (8 claims)", "Customer profile", "Segments", "Silence risk",
  "Adoption", "Alerts (50 ranked)", "Simulator", "Methods", "Overview". No "Home", no "Dashboard".
- Copy is plain and specific; sentence case; no marketing adjectives; no exclamation marks.
- Light and dark themes with equal care; test both at 1440px and at 390px.

## 5. The pages

- **Overview**: a left column of four or five numbers with intervals and a one-line context each,
  a right column with the channel mix and the calendar heatmap; no KPI tiles.
- **Findings**: one card per claim: the question in plain words as the title, the value with its
  interval, the robustness badge, a one-sentence mechanism, and the n and period in a caption.
- **Customer profile**: search at the top, then profile, timeline, segment, silence risk with its
  three reasons, and open alerts, as one scrolling page with section anchors.
- **Segments**: the UMAP scatter with the profile table beside it; hovering a point highlights its
  segment's row; the stability number sits in the table header.
- **Silence risk**: the ranked table first, the calibration curve and lift chart second.
- **Adoption**: the decile table with lift, then the gain chart.
- **Alerts**: a ranked list with rule evidence in a drawer that slides from the right and returns
  the same way; feedback buttons change state in place.
- **Simulator**: sliders left, bands right, the policy sentence written in words above the chart
  ("Move 30% of branch-only clients under 45 to mixed use").
- **Methods**: equations with prose between them, at a reading measure.

## 6. Deliverables

A tokens file (`src/styles/tokens.css`) with light and dark values; every component restyled to
use only tokens; the ECharts theme generated from the tokens; Storybook stories in both themes;
Playwright screenshots of every page in both themes at 1440px and at 390px replacing the images
under `docs/images/`; the axe checks and the bundle check still green; a short
`docs/DESIGN_NOTES.md` listing every decision that departs from this brief and why.
