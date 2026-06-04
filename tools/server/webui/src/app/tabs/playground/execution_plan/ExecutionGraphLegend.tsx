import type { InkPalette } from "../../../../components/plotTokens";
import { kindColor, kindLabel, type ExecKindKey } from "./mathbox/executionScene";

export function ExecutionGraphLegend({
  kinds,
  palette,
}: {
  kinds: ExecKindKey[];
  palette: InkPalette;
}) {
  if (!kinds.length) return null;
  return (
    <ul className="exec-graph__legend" aria-label="Kernel partition kinds">
      {kinds.map((kind) => (
        <li key={kind} className="exec-graph__legend-item">
          <span
            className="exec-graph__legend-swatch"
            style={{ backgroundColor: kindColor(kind, palette) }}
            aria-hidden="true"
          />
          {kindLabel(kind)}
        </li>
      ))}
    </ul>
  );
}
