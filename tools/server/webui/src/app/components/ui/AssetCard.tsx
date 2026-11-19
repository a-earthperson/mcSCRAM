import type { PlotAsset } from "../../types";
import { formatBytes } from "../../lib/format";

export function AssetCard({ asset, onPreview }: { asset: PlotAsset; onPreview: (asset: PlotAsset) => void }) {
  const assetUrl = `${import.meta.env.BASE_URL}${asset.assetPath}`;

  return (
    <article className="asset-card">
      <a
        className="asset-card-link"
        href={assetUrl}
        onClick={(event) => {
          event.preventDefault();
          onPreview(asset);
        }}
        aria-label={`Preview ${asset.title}`}
      >
        <img src={assetUrl} alt={asset.title} loading="lazy" />
        <div>
          <strong>{asset.title}</strong>
          <span>
            {asset.category} · {asset.extension.toUpperCase()} · {formatBytes(asset.sizeBytes)}
          </span>
        </div>
      </a>
    </article>
  );
}
