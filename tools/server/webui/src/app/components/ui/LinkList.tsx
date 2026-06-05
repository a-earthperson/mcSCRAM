import { formatBytes } from "../../lib/format";

export function LinkList({ artifacts }: { artifacts: Array<{ title: string; path: string; url: string; extension: string; sizeBytes?: number; kind?: string }> }) {
  return (
    <ul className="link-list">
      {artifacts.map((artifact) => (
        <li key={artifact.path}>
          <a href={artifact.url}>{artifact.title}</a>
          <span>
            {artifact.kind ?? artifact.extension.toUpperCase()} · {artifact.path}
            {artifact.sizeBytes ? ` · ${formatBytes(artifact.sizeBytes)}` : ""}
          </span>
        </li>
      ))}
    </ul>
  );
}
