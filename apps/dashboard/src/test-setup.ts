import "@testing-library/jest-dom/vitest";

// jsdom has no ResizeObserver; Radix primitives measure their thumbs with one.
class ResizeObserverStub {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}
globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver;
