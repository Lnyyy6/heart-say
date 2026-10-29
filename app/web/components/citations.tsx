import type { Citation } from "@/lib/types";
import { IconArrowUpRight, IconChevron } from "./icons";

export function Citations({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <details className="citations">
      <summary>
        <span className="citation-ref">REF</span>
        <span>{citations.length} 则知识库参考</span>
        <IconChevron className="citation-chevron" />
      </summary>
      <div className="citation-list">
        {citations.map((citation) => {
          const external = citation.source_url?.startsWith("http");
          const content = (
            <>
              <span className="citation-index">{String(citation.index).padStart(2, "0")}</span>
              <span className="citation-body">
                <span className="citation-title">
                  {citation.title}
                  {external && <IconArrowUpRight className="citation-external" />}
                </span>
                <span className="citation-snippet">{citation.snippet}</span>
                <span className="citation-source">
                  {citation.source_name}
                  <span className="citation-score">
                    {(citation.rerank_score ?? citation.score).toFixed(2)}
                  </span>
                </span>
              </span>
            </>
          );
          return external ? (
            <a
              className="citation-item"
              href={citation.source_url}
              target="_blank"
              rel="noreferrer noopener"
              key={citation.doc_id}
            >
              {content}
            </a>
          ) : (
            <div className="citation-item" key={citation.doc_id}>
              {content}
            </div>
          );
        })}
      </div>
    </details>
  );
}
