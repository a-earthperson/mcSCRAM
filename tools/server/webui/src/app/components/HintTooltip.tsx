import * as Tooltip from "@radix-ui/react-tooltip";
import type { ReactNode } from "react";
import "../styles/hint-tooltip.css";

export function TooltipProvider({ children }: { children: ReactNode }) {
  return (
    <Tooltip.Provider delayDuration={280} skipDelayDuration={120}>
      {children}
    </Tooltip.Provider>
  );
}

export function HintTooltip({
  content,
  children,
}: {
  content: ReactNode;
  children: ReactNode;
}) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content className="hint-tooltip" side="top" sideOffset={6} collisionPadding={8}>
          <div className="hint-tooltip__content">{content}</div>
          <Tooltip.Arrow className="hint-tooltip__arrow" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

export function FieldHintTrigger({ label, content }: { label: string; content: ReactNode }) {
  return (
    <HintTooltip content={content}>
      <button type="button" className="field-hint-trigger" aria-label={label}>
        ?
      </button>
    </HintTooltip>
  );
}
