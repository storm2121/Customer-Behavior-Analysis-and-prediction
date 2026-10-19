import { useEffect, useState } from "react";

/**
 * The width of an element, for chart layouts that change shape on a phone (not only size). The
 * ref is a callback, so an element that mounts later (after its data arrives) is measured too.
 */
export function useElementWidth<T extends HTMLElement>(): [(node: T | null) => void, number] {
  const [node, setNode] = useState<T | null>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    if (!node) return;
    const measure = () => setWidth(node.getBoundingClientRect().width);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, [node]);
  return [setNode, width];
}

/**
 * Which edges of a horizontal scroller have more content beyond them, for the soft edge fades
 * (`.scroll-fade-x`). The ref is a callback, so a scroller that mounts later is measured too.
 */
export function useScrollEdges<T extends HTMLElement>(): [(node: T | null) => void, { before: boolean; after: boolean }] {
  const [node, setNode] = useState<T | null>(null);
  const [more, setMore] = useState({ before: false, after: false });
  useEffect(() => {
    if (!node) return;
    const measure = () => {
      const before = node.scrollLeft > 1;
      const after = node.scrollLeft + node.clientWidth < node.scrollWidth - 1;
      setMore((current) => (current.before === before && current.after === after ? current : { before, after }));
    };
    measure();
    node.addEventListener("scroll", measure, { passive: true });
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => {
      node.removeEventListener("scroll", measure);
      observer.disconnect();
    };
  }, [node]);
  return [setNode, more];
}

/** `value`, once it has stopped changing for `delay` milliseconds (for network calls, never visuals). */
export function useDebounced<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delay);
    return () => window.clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}
