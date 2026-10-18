import "katex/dist/katex.min.css";

import katex from "katex";
import { useMemo } from "react";

/** A display equation rendered by KaTeX (the source is the project's own methods, not data). */
export function Equation({ tex, label }: { tex: string; label: string }) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: true, throwOnError: false, output: "htmlAndMathml" }),
    [tex],
  );
  // A wide equation scrolls sideways on a phone; the tab stop lets keyboard users scroll it too.
  return (
    <div
      role="group"
      aria-label={label}
      tabIndex={0}
      className="overflow-x-auto rounded-md text-ink"
      // KaTeX output of a constant string from this file's callers, never of data.
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
