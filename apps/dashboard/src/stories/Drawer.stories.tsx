import type { Meta, StoryObj } from "@storybook/react";
import { PanelRightOpen } from "lucide-react";
import { useState } from "react";

import { Drawer } from "../components/Drawer";
import { Button } from "../components/Pressable";

/** The evidence drawer: it slides in from the right, leaves the same way, and can be reversed half-way. */
function DrawerDemo() {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <Button icon={PanelRightOpen} onClick={() => setOpen(true)}>
        Evidence
      </Button>
      <Drawer open={open} onOpenChange={setOpen} title="Alert 1" description="Client syn-example, illustrative content">
        <p className="text-small text-ink">The rules that fired and their evidence appear here, with the review buttons.</p>
      </Drawer>
    </div>
  );
}

const meta = {
  title: "Overlays/Drawer",
  component: DrawerDemo,
} satisfies Meta<typeof DrawerDemo>;

export default meta;
type Story = StoryObj<typeof meta>;

export const EvidenceDrawer: Story = {};
