# Design notes: where the dashboard departs from the brief

The dashboard follows [the design brief](DESIGN.md). This page lists every decision that departs
from it, or reads it in a way it does not spell out, and why. Everything else is as the brief says.
The tokens and their rationale are in `apps/dashboard/src/styles/tokens.css`; a unit test keeps the
computable promises (contrast, the palette's lightness and colour-vision separation, the two dark
blocks agreeing).

## Data and honesty

- **The Findings page is not in the screenshots.** The brief asks for every page. Findings shows the
  allow-listed claims of the real extract; in a picture those values would escape
  `strata ledger check` and go stale when the ledger is republished, so they stay in the ledger and
  the documents rendered from it, as the README has said since the documentation milestone. The
  capture script takes the other pages in both themes at both widths.
- **The data label on Findings says what that page shows.** The brief keeps "synthetic demo" on every
  page "since the public stack never shows real data", but Findings does show real, allow-listed
  aggregates. The top bar keeps "Synthetic demo" everywhere and adds "Claims: reference extract" on
  Findings, with a notice on the page itself, so a real result is never read as synthetic (or the
  reverse).
- **The API gained context, not behaviour.** To give every number its interval, sample and period,
  the API now serves, beside what it already served: each public claim's method, caveats,
  robustness explanation, event count, period and source; the overview's period and the zone
  ledger's channel-mix intervals (only when that ledger was built on the data in the warehouse); the
  segments' embedding and client count; the size of the scored population; and the description of
  every interval it serves (model metrics, segment stability, simulator bands). The descriptions are
  defined once beside the percentiles that produce them, so the dashboard never types a coverage
  level into its copy.
- **A claim card lists values, not one value.** Each public claim holds several values and the ledger
  names no headline among them, so the card lists the values that carry an interval first, in one
  format, with the rest behind a disclosure. The "one-sentence mechanism" is the robustness
  checker's explanation where a claim has a verdict (the ledger's own text, sometimes longer than a
  sentence); a claim without a verdict shows its method instead, since the ledger records no
  mechanism for it.

## Colour and charts

- **Robustness verdicts are green, amber and violet.** The brief fixes three colours without naming
  them. A reversal is drawn in violet rather than a danger red: a headline that reverses under
  stratification is a finding the system exists to report, not an error, and green-to-red would be
  the red-to-green scale the brief rules out. Each badge also carries its word and its icon.
- **The segment map uses the neutral ramp and the accent, not the categorical hues.** BIC can choose
  more segments than the categorical palette has hues, and a scatter puts any two segments side by
  side, a separation that equal-lightness hues cannot give across a whole palette. One segment is in
  the accent at a time; pointing at a dot or at a row of the table chooses it.
- **The ECharts theme is merged into each option rather than registered.** It is generated from the
  tokens on each chart's own element, which keeps one chart instance for life: ECharts 5 binds a
  registered theme when a chart is created, so switching it would remount every chart and replay
  its entrance.
- **The stability number sits in the header of the profile panel**, directly above the table, rather
  than inside the table's own header row, where it would scroll sideways with the table on a phone.

## Type and shape

- **Section headings use a looser leading than titles.** Page titles and the large figures keep the
  brief's title leading; the smaller section headings, which include the claim cards' questions and
  often wrap, use a slightly looser one so wrapped lines do not collide.
- **Equations are set flush left**, with a small indent, instead of KaTeX's centred display, in line
  with the brief's rule against centred layouts.
- **Two small controls carry a shadow.** The slider thumb and the selected segment of a segmented
  control sit on top of their track; they are read as floating surfaces in the brief's sense, and
  carry the smallest elevation token. In-flow panels have none.

## Motion and interaction

- **The simulator redraws by interpolation while a slider moves.** The brief asks for continuous
  redraws with the network call debounced. Between policies the API has already computed, the bands
  are interpolated linearly in the share moved: exact for every posterior draw (the projection is
  linear in the starting counts), close but not exact for the quantiles across draws. The bands of
  the value a slider settles on are always the API's own, and the page says so. The age limit moves
  in whole bands, so its bands switch as the thumb crosses each step. Interpolation anchors are
  fetched after the first answer, and for every age limit once the age slider is first touched.
- **Releasing a slider settles on the nearest step from where it was let go**, with the release
  velocity handed to the spring, but without projecting the momentum forward: on a value control,
  the place the reader lets go is the value they meant.
- **Feedback on an alert is given in the evidence drawer**, beside the evidence it rests on, and the
  buttons change state there; each row of the queue shows the review state. The queue's default
  budget is unchanged, so its title counts the alerts shown under the budget chosen.
- **While a sheet is open the top bar turns solid**, so there is never more than one translucent
  material on screen, as the brief asks.
- **A theme change is instant.** Only transform and opacity animate, so there is no cross-fade of
  colours between themes.
