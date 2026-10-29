"use client";

import { useEffect, useRef } from "react";
import { IconArrowUp, IconStop } from "./icons";

export function Composer({
  value,
  onChange,
  onSend,
  onStop,
  streaming,
}: {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  streaming: boolean;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const composingRef = useRef(false);

  useEffect(() => {
    const node = textareaRef.current;
    if (!node) return;
    node.style.height = "0px";
    node.style.height = `${Math.min(node.scrollHeight, 168)}px`;
  }, [value]);

  const submit = () => {
    if (streaming || !value.trim()) return;
    onSend();
  };

  return (
    <form
      className="composer"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      <div className={`composer-box ${streaming ? "is-streaming" : ""}`}>
        <label htmlFor="heart-say-input" className="sr-only">
          说说你现在想聊的事
        </label>
        <textarea
          id="heart-say-input"
          ref={textareaRef}
          rows={1}
          value={value}
          maxLength={4000}
          placeholder="说说此刻最让你放不下的事…"
          onChange={(event) => onChange(event.target.value)}
          onCompositionStart={() => {
            composingRef.current = true;
          }}
          onCompositionEnd={() => {
            composingRef.current = false;
          }}
          onKeyDown={(event) => {
            if (
              event.key === "Enter" &&
              !event.shiftKey &&
              !event.nativeEvent.isComposing &&
              !composingRef.current
            ) {
              event.preventDefault();
              submit();
            }
          }}
        />
        <div className="composer-actions">
          {value.length > 1200 && (
            <span className="composer-count">{value.length} / 4000</span>
          )}
          {streaming ? (
            <button
              type="button"
              className="send-button stop-button"
              onClick={onStop}
              aria-label="停止生成"
            >
              <IconStop />
            </button>
          ) : (
            <button
              type="submit"
              className="send-button"
              disabled={!value.trim()}
              aria-label="发送"
            >
              <IconArrowUp />
            </button>
          )}
        </div>
      </div>
      <div className="composer-foot">
        <span>Enter 发送 · Shift + Enter 换行</span>
        <span>心语提供心理教育支持，不替代诊断或治疗</span>
      </div>
    </form>
  );
}
