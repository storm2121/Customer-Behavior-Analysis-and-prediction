// The tokens file is the design's single source; these checks keep its promises computable:
// the two dark blocks agree, every token the chart theme reads exists, text keeps WCAG AA on the
// surfaces it sits on, and the categorical palette keeps equal lightness and the colour-vision
// separation it was chosen for (the same OKLab and Machado simulation as the dataviz validator).

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { CHART_TOKEN_NAMES } from "../charts/theme";

// Read from disk: Vitest does not process stylesheets, so a ?raw import would come back empty.
const css = readFileSync(resolve(__dirname, "tokens.css"), "utf8").replace(/\r\n/g, "\n");

function block(selector: string): Record<string, string> {
  const start = css.indexOf(`${selector} {`);
  if (start < 0) throw new Error(`no block for ${selector}`);
  const body = css.slice(css.indexOf("{", start) + 1, css.indexOf("}", start));
  const out: Record<string, string> = {};
  for (const match of body.matchAll(/--([\w-]+):\s*([^;]+);/g)) out[match[1]!] = match[2]!.trim();
  return out;
}

const light = block(':root,\n[data-theme="light"]');
const darkFromSystem = block(':root:not([data-theme="light"])');
const darkChosen = block('[data-theme="dark"]');
const semantic = block(":root,\n[data-theme]");

type Theme = Record<string, string>;

/** A token's colour in a theme, following var() references through the semantic layer. */
function colour(theme: Theme, name: string): string {
  const value = theme[name] ?? semantic[name];
  if (!value) throw new Error(`no token --${name}`);
  const reference = /^var\(--([\w-]+)\)$/.exec(value);
  if (reference) return colour(theme, reference[1]!);
  if (!/^#[0-9a-f]{6}$/i.test(value)) throw new Error(`--${name} is not a plain colour: ${value}`);
  return value;
}

const channels = (hex: string) => [1, 3, 5].map((i) => Number.parseInt(hex.slice(i, i + 2), 16) / 255);
const linear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const luminance = (hex: string) => {
  const [r, g, b] = channels(hex).map(linear) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
function contrast(a: string, b: string): number {
  const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number];
  return (high + 0.05) / (low + 0.05);
}
/** `share` of `top` over `base`, as a tint made with color-mix or an opacity modifier. */
function mix(top: string, base: string, share: number): string {
  const [t, b] = [channels(top), channels(base)];
  return `#${t.map((c, i) => Math.round(255 * (share * c + (1 - share) * b[i]!)).toString(16).padStart(2, "0")).join("")}`;
}

function oklab(rgbLinear: number[]): [number, number, number] {
  const [r, g, b] = rgbLinear as [number, number, number];
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}
// Machado, Oliveira and Fernandes (2009), severity 1, on linear RGB.
const MACHADO: Record<string, number[][]> = {
  protan: [
    [0.152286, 1.052583, -0.204868],
    [0.114503, 0.786281, 0.099216],
    [-0.003882, -0.048116, 1.051998],
  ],
  deutan: [
    [0.367322, 0.860646, -0.227968],
    [0.280085, 0.672501, 0.047413],
    [-0.01182, 0.04294, 0.968881],
  ],
};
function seen(hex: string, kind?: string): [number, number, number] {
  const rgb = channels(hex).map(linear);
  if (!kind) return oklab(rgb);
  const matrix = MACHADO[kind]!;
  return oklab(matrix.map((row) => Math.min(1, Math.max(0, row[0]! * rgb[0]! + row[1]! * rgb[1]! + row[2]! * rgb[2]!))));
}
function deltaE(a: string, b: string, kind?: string): number {
  const [x, y] = [seen(a, kind), seen(b, kind)];
  return 100 * Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]);
}

const THEMES: [string, Theme][] = [
  ["light", light],
  ["dark", darkChosen],
];

describe("the design tokens", () => {
  it("define the same dark values for the system preference and for a chosen dark theme", () => {
    expect(darkFromSystem).toEqual(darkChosen);
  });

  it("give every base token a value in both themes", () => {
    expect(Object.keys(darkChosen).sort()).toEqual(Object.keys(light).sort());
  });

  it("define every token the chart theme reads", () => {
    for (const name of CHART_TOKEN_NAMES) expect(css, `--${name}`).toContain(`--${name}:`);
  });

  it.each(THEMES)("keep text at WCAG AA on the surfaces it sits on (%s)", (_, theme) => {
    const c = (name: string) => colour(theme, name);
    const surfaces = ["canvas", "surface", "surface-raised", "grey-1", "grey-2"].map(c);
    const pairs: [string, string[]][] = [
      ["ink", [...surfaces, mix(c("accent"), c("surface"), 0.1)]],
      ["grey-7", surfaces],
      ["grey-8", surfaces],
      ["accent", [c("canvas"), c("surface"), c("surface-raised"), c("grey-1"), mix(c("accent"), c("surface"), 0.1)]],
      ["success", [c("surface"), c("canvas"), mix(c("success"), c("surface"), 0.1)]],
      ["warning", [c("surface"), c("canvas"), mix(c("warning"), c("surface"), 0.1)]],
      ["danger", [c("surface"), c("canvas"), mix(c("danger"), c("surface"), 0.1)]],
      ["verdict-holds", [c("surface"), mix(c("verdict-holds"), c("surface"), 0.1)]],
      ["verdict-weakens", [c("surface"), mix(c("verdict-weakens"), c("surface"), 0.1)]],
      ["verdict-reverses", [c("surface"), mix(c("verdict-reverses"), c("surface"), 0.1)]],
    ];
    for (const [text, backgrounds] of pairs) {
      for (const background of backgrounds) {
        expect(contrast(c(text), background), `${text} on ${background}`).toBeGreaterThanOrEqual(4.5);
      }
    }
    expect(contrast(c("on-accent"), c("accent-fill")), "on-accent on accent-fill").toBeGreaterThanOrEqual(4.5);
  });

  it.each(THEMES)("keep marks, the focus ring and the neutral series at 3:1 on the chart surface (%s)", (_, theme) => {
    const c = (name: string) => colour(theme, name);
    for (const name of ["chart-1", "chart-2", "chart-3", "chart-4", "chart-5", "chart-neutral", "accent"]) {
      expect(contrast(c(name), c("surface")), name).toBeGreaterThanOrEqual(3);
      expect(contrast(c(name), c("canvas")), `${name} on canvas`).toBeGreaterThanOrEqual(3);
    }
  });

  it.each(THEMES)("hold the five categorical hues at equal lightness, separable in colour-vision deficiency (%s)", (_, theme) => {
    const palette = ["chart-1", "chart-2", "chart-3", "chart-4", "chart-5"].map((name) => colour(theme, name));
    const lightness = palette.map((hex) => seen(hex)[0]);
    expect(Math.max(...lightness) - Math.min(...lightness)).toBeLessThan(0.02);
    const pairs = (indexes: number[]) => indexes.flatMap((i, k) => indexes.slice(k + 1).map((j) => [i, j] as const));
    const adjacent = palette.slice(1).map((_, i) => [i, i + 1] as const);
    for (const [i, j] of [...adjacent, ...pairs([0, 1, 2])]) {
      const [a, b] = [palette[i]!, palette[j]!];
      expect(Math.min(deltaE(a, b, "protan"), deltaE(a, b, "deutan")), `${a} and ${b} under CVD`).toBeGreaterThanOrEqual(8);
      expect(deltaE(a, b), `${a} and ${b}`).toBeGreaterThanOrEqual(15);
    }
  });
});
