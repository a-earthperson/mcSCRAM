import { useEffect } from "react";
import type { PlotAsset } from "../../types";

export function AssetPreviewModal({ asset, onClose }: { asset: PlotAsset; onClose: () => void }) {
  const assetUrl = `${import.meta.env.BASE_URL}${asset.assetPath}`;

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    document.body.classList.add("modal-open");
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.classList.remove("modal-open");
    };
  }, [onClose]);

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section
        className="image-modal"
        role="dialog"
        aria-modal="true"
        aria-label={`${asset.title} preview`}
        onClick={(event) => event.stopPropagation()}
      >
        <header>
          <div>
            <p className="eyebrow">{asset.category}</p>
            <h2>{asset.title}</h2>
            <p>{asset.source}</p>
          </div>
          <button className="modal-close" type="button" onClick={onClose} aria-label="Close preview">
            Close
          </button>
        </header>
        <div className="modal-image-frame">
          <img src={assetUrl} alt={asset.title} />
        </div>
        <footer>
          <a href={assetUrl}>Open bundled image</a>
          <a href={asset.sourceUrl}>View source path on GitHub</a>
        </footer>
      </section>
    </div>
  );
}
