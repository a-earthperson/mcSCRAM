import * as THREE from "three";
import type { InkPalette } from "../../../../components/plot/plotTokens";
import type { MathBoxView } from "./types";

export const AXIS_LABEL_COLORS_ID = "exec-axis-label-colors";

function paletteColorRgba(color: string): [number, number, number, number] {
  const c = new THREE.Color(color);
  return [c.r, c.g, c.b, 1];
}

/** RGB axis lines and x/y/z labels in the active cartesian view. */
export function buildGlobalAxes(view: MathBoxView, palette: InkPalette): void {
  const axisStyle = {
    range: [-1, 1] as [number, number],
    width: 4,
    end: true,
    start: false,
    opacity: 0.92,
    zBias: -12,
    zWrite: false,
  };

  view.axis({ ...axisStyle, axis: 1, color: palette.madder });
  view.axis({ ...axisStyle, axis: 2, color: palette.sage });
  view.axis({ ...axisStyle, axis: 3, color: palette.indigo });

  view.array({
    id: AXIS_LABEL_COLORS_ID,
    live: false,
    data: [
      paletteColorRgba(palette.madder),
      paletteColorRgba(palette.sage),
      paletteColorRgba(palette.indigo),
    ],
  });

  view
    .array({
      data: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      channels: 3,
      live: false,
    })
    .text({
      data: ["x", "y", "z"],
    })
    .label({
      colors: `#${AXIS_LABEL_COLORS_ID}`,
      color: palette.ink,
      size: 14,
      zBias: -11,
      zWrite: false,
    });
}
