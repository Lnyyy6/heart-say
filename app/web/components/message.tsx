import type { UiMessage } from "@/lib/types";
import { BrandMark } from "./brand";
import { Citations } from "./citations";
import { IconRefresh } from "./icons";

export function Message({
  message,
  isLast,
  onRetry,
}: {
  message: UiMessage;
  isLast: boolean;
  onRetry?: () => void;
}) {
  const assistant = message.role === "assistant";
  const total = message.trace?.timings_ms?.total_ms;

  return (
    <article className={`message message-${message.role}`}>
      {assistant && (
        <div className="message-mark" aria-hidden="true">
          <BrandMark size={22} />
        </div>
      )}
      <div className="message-body">
        {assistant && (
          <div className="message-label">
            <span>心语</span>
            <span className="message-label-divider" />
            <span className="message-label-muted">
              {message.streaming ? "正在回应" : "陪伴"}
            </span>
          </div>
        )}
        <div
          className={`message-text ${message.streaming ? "is-streaming" : ""}`}
          aria-live={assistant && message.streaming ? "polite" : undefined}
        >
          {message.content ? (
            message.content
          ) : message.streaming ? (
            <span className="thinking">正在想怎么接住你</span>
          ) : null}
          {message.streaming && message.content && <span className="stream-cursor" aria-hidden="true" />}
        </div>

        {message.error && (
          <div className="message-error">
            <span>{message.error}</span>
            {isLast && onRetry && (
              <button type="button" className="text-action" onClick={onRetry}>
                <IconRefresh className="action-icon" />
                再试一次
              </button>
            )}
          </div>
        )}

        {message.stopped && !message.error && (
          <div className="message-stopped">
            <span>已停止生成</span>
            {isLast && onRetry && (
              <button type="button" className="text-action" onClick={onRetry}>
                <IconRefresh className="action-icon" />
                重新生成
              </button>
            )}
          </div>
        )}

        {!!message.citations?.length && <Citations citations={message.citations} />}

        {assistant && message.trace && !message.streaming && !message.error && (
          <div className="message-trace">
            <span>
              {message.safety?.action === "protocol"
                ? "安全协议 · 未检索"
                : message.trace.use_rag
                  ? `知识库 ${message.trace.retrieved.length} 则`
                  : "直接回应"}
            </span>
            {typeof total === "number" && <span>{Math.round(total)} ms</span>}
            {!!message.trace.tool_calls?.length && (
              <span>工具 {message.trace.tool_calls.length}</span>
            )}
            <span className="trace-graph">{message.trace.graph.at(-1)}</span>
          </div>
        )}
      </div>
    </article>
  );
}
