import * as Dialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, type ReactNode } from "react";

import { CROSS_FADE, SPRING, useCalmMotion } from "../motion";
import { IconButton } from "./Pressable";

/**
 * A sheet that slides in from the right and leaves the same way. Radix gives it the dialog's
 * focus handling (focus moves in, Escape closes, focus returns to the control that opened it);
 * Motion animates it from wherever it is, so closing it half-way in reverses it smoothly. While it
 * is open the top bar turns solid: one translucent material at a time. Under reduced motion it
 * cross-fades instead of sliding.
 */
export function Drawer({
  open,
  onOpenChange,
  title,
  description,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
  children: ReactNode;
}) {
  const calm = useCalmMotion();
  // What had focus when the drawer opened gets it back when the drawer closes (Radix would send it
  // to a Dialog.Trigger, which a drawer opened from a list row does not have). Read during render,
  // before the dialog moves focus inside itself.
  const opener = useRef<HTMLElement | null>(null);
  const wasOpen = useRef(false);
  if (open && !wasOpen.current && typeof document !== "undefined") opener.current = document.activeElement as HTMLElement | null;
  wasOpen.current = open;

  useEffect(() => {
    if (!open) return;
    const root = document.documentElement;
    root.setAttribute("data-sheet-open", "");
    return () => root.removeAttribute("data-sheet-open");
  }, [open]);

  const slide = calm
    ? { initial: { opacity: 0 }, animate: { opacity: 1 }, exit: { opacity: 0 }, transition: CROSS_FADE }
    : { initial: { x: "100%" }, animate: { x: 0 }, exit: { x: "100%" }, transition: SPRING };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <AnimatePresence>
        {open ? (
          <Dialog.Portal forceMount>
            <Dialog.Overlay asChild forceMount>
              <motion.div
                className="fixed inset-0 z-40 bg-scrim"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={calm ? CROSS_FADE : SPRING}
              />
            </Dialog.Overlay>
            <Dialog.Content
              asChild
              forceMount
              onCloseAutoFocus={(event) => {
                event.preventDefault();
                opener.current?.focus();
              }}
            >
              <motion.div
                {...slide}
                className="material fixed inset-y-0 right-0 z-50 flex w-[var(--drawer-width)] max-w-full flex-col shadow-float outline-none sm:inset-y-2 sm:right-2 sm:rounded-sheet"
              >
                <header className="flex items-start justify-between gap-3 px-5 pt-4 pb-3">
                  <div className="flex min-w-0 flex-col gap-1">
                    <Dialog.Title className="text-heading font-title text-ink">{title}</Dialog.Title>
                    <Dialog.Description className="text-small text-ink-secondary">{description}</Dialog.Description>
                  </div>
                  <Dialog.Close asChild>
                    <IconButton label="Close" icon={X} />
                  </Dialog.Close>
                </header>
                <div className="min-h-0 flex-1 overflow-y-auto px-5 pb-6">{children}</div>
              </motion.div>
            </Dialog.Content>
          </Dialog.Portal>
        ) : null}
      </AnimatePresence>
    </Dialog.Root>
  );
}
