import type {
  ChatResponse,
  Citation,
  EvalSummaryResponse,
  Meta,
  SafetyInfo,
  SessionDetail,
  TraceInfo,
} from "./types";

const BASE = (process.env.NEXT_PUBLIC_API_BASE || "/api").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;

  constructor(message: string, status = 0) {
    super(message);
    this.status = status;
  }
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!response.ok) {
    throw new ApiError(`请求失败：${response.status}`, response.status);
  }
  return (await response.json()) as T;
}

export function getMeta(): Promise<Meta> {
  return getJson<Meta>("/meta");
}

export function getSession(sessionId: string): Promise<SessionDetail> {
  return getJson<SessionDetail>(`/sessions/${encodeURIComponent(sessionId)}`);
}

export function getEvalSummary(): Promise<EvalSummaryResponse> {
  return getJson<EvalSummaryResponse>("/eval/summary");
}

export interface StreamHandlers {
  onMeta?: (payload: { session_id: string; request_id: string; safety: SafetyInfo; trace: TraceInfo }) => void;
  onDelta?: (text: string) => void;
  onCitations?: (citations: Citation[]) => void;
  onDone?: (response: ChatResponse) => void;
  onError?: (message: string) => void;
}

function parseSseBlock(raw: string): { event: string; data: string } | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of raw.replace(/\r/g, "").split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (!dataLines.length) return null;
  return { event, data: dataLines.join("\n") };
}

export async function streamChat(
  payload: { message: string; session_id: string; use_rag: boolean },
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ ...payload, stream: true }),
      signal,
    });
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new ApiError("无法连接心语服务，请确认 API 已启动。");
  }

  if (!response.ok || !response.body) {
    throw new ApiError(`心语服务返回 ${response.status}，请稍后重试。`, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const handle = (block: string) => {
    const parsed = parseSseBlock(block);
    if (!parsed) return;
    let data: unknown;
    try {
      data = JSON.parse(parsed.data);
    } catch {
      return;
    }
    if (parsed.event === "meta") {
      handlers.onMeta?.(
        data as { session_id: string; request_id: string; safety: SafetyInfo; trace: TraceInfo },
      );
    } else if (parsed.event === "delta") {
      handlers.onDelta?.((data as { text: string }).text ?? "");
    } else if (parsed.event === "citations") {
      handlers.onCitations?.(data as Citation[]);
    } else if (parsed.event === "done") {
      handlers.onDone?.(data as ChatResponse);
    } else if (parsed.event === "error") {
      handlers.onError?.((data as { message: string }).message ?? "生成失败，请重试。");
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // sse-starlette 使用 CRLF；统一后再按空行切事件
    buffer = buffer.replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      handle(block);
      boundary = buffer.indexOf("\n\n");
    }
  }
  buffer += decoder.decode();
  buffer = buffer.replace(/\r\n/g, "\n");
  if (buffer.trim()) handle(buffer);
}
