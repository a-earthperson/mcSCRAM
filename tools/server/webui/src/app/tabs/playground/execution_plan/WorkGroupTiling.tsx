import { useMemo } from "react";
import type { InkPalette } from "../../../../components/plotTokens";
import { withAlpha } from "../../../../components/plotTokens";
import { compactCount, formatRange3, type ExecKindKey, kindColor } from "./mathbox/executionScene";
import type { ExecutionPartitionDescription } from "../../../faultTreeApi";

/**
 * Shows how a partition's global launch range is tiled into work-groups by its
 * local range. The grid is the device's chunking decision made literal: each
 * cell is one work-group of `local` work-items. Counts are exact; the drawn
 * grid is truncated (with an ellipsis) when the launch has more groups than can
 * be legibly shown.
 */

const MAX_COLS = 12;
const MAX_ROWS = 8;
const CELL = 16;
const GAP = 3;
const PAD = 2;

function groups(global: number, local: number): number {
  if (!local) return global > 0 ? 1 : 0;
  return Math.max(1, Math.round(global / local));
}

export function WorkGroupTiling({
  partition,
  colorKey,
  palette,
}: {
  partition: ExecutionPartitionDescription;
  colorKey: ExecKindKey;
  palette: InkPalette;
}) {
  const model = useMemo(() => {
    const wgX = groups(partition.global_range.x, partition.local_range.x);
    const wgY = groups(partition.global_range.y, partition.local_range.y);
    const wgZ = groups(partition.global_range.z, partition.local_range.z);
    const localSize =
      partition.local_range.x * partition.local_range.y * partition.local_range.z;
    const cols = Math.max(1, wgX);
    const rows = Math.max(1, wgZ);
    const visCols = Math.min(cols, MAX_COLS);
    const visRows = Math.min(rows, MAX_ROWS);
    return { wgX, wgY, wgZ, localSize, cols, rows, visCols, visRows };
  }, [partition]);

  const color = kindColor(colorKey, palette);
  const width = PAD * 2 + model.visCols * CELL + (model.visCols - 1) * GAP;
  const height = PAD * 2 + model.visRows * CELL + (model.visRows - 1) * GAP;
  const truncatedX = model.cols > model.visCols;
  const truncatedZ = model.rows > model.visRows;

  const cells: Array<{ x: number; y: number; faded: boolean }> = [];
  for (let row = 0; row < model.visRows; row += 1) {
    for (let col = 0; col < model.visCols; col += 1) {
      const faded =
        (truncatedX && col === model.visCols - 1) || (truncatedZ && row === model.visRows - 1);
      cells.push({
        x: PAD + col * (CELL + GAP),
        y: PAD + row * (CELL + GAP),
        faded,
      });
    }
  }

  return (
    <div className="exec-tiling">
      <div className="exec-tiling__head">
        <span className="exec-tiling__title">work-group tiling</span>
        <span className="exec-tiling__formula">
          {formatRange3(partition.global_range)} &divide; {formatRange3(partition.local_range)}
        </span>
      </div>
      <div className="exec-tiling__body">
        <svg
          className="exec-tiling__svg"
          viewBox={`0 0 ${width} ${height}`}
          width={width}
          height={height}
          role="img"
          aria-label={`${model.wgX} by ${model.wgZ} work-group tiling`}
        >
          {cells.map((cell, index) => (
            <rect
              key={index}
              x={cell.x}
              y={cell.y}
              width={CELL}
              height={CELL}
              rx={2}
              fill={withAlpha(color, cell.faded ? 0.12 : 0.32)}
              stroke={withAlpha(color, cell.faded ? 0.3 : 0.9)}
              strokeWidth={1}
            />
          ))}
        </svg>
        <dl className="exec-tiling__facts">
          <div>
            <dt>work-groups</dt>
            <dd>
              {compactCount(model.wgX)} x {compactCount(model.wgY)} x {compactCount(model.wgZ)}
            </dd>
          </div>
          <div>
            <dt>group size</dt>
            <dd>{compactCount(model.localSize)} wi</dd>
          </div>
          <div>
            <dt>total groups</dt>
            <dd>{compactCount(model.cols * model.rows * model.wgY)}</dd>
          </div>
        </dl>
      </div>
      {(truncatedX || truncatedZ) ? (
        <p className="exec-tiling__note">Grid truncated &mdash; full launch is {compactCount(model.wgX)} x {compactCount(model.wgZ)} groups.</p>
      ) : null}
    </div>
  );
}
