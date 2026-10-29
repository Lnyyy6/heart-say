export type RiskLevel = "none" | "low" | "medium" | "high" | "critical";
export type Role = "user" | "assistant";

export interface Citation {
  index: number;
  doc_id: string;
  title: string;
  topic: string;
  source_name: string;
  source_url: string;
  snippet: string;
  score: number;
  rerank_score: number | null;
}

export interface SafetyInfo {
  level: RiskLevel;
  categories: string[];
  action: string;
  disclaimer: string;
  resources: { name: string; contact: string; description: string }[];
  guard_flags: string[];
}

export interface TraceInfo {
  request_id: string;
  model: string;
  backend: string;
  use_rag: boolean;
  audience: string;
  emotion: string;
  intent: string;
  timings_ms: Record<string, number>;
  retrieved: { id: string; title: string; score: number; rerank_score: number | null }[];
  tool_calls?: { name: string; output: string }[];
  graph: string[];
}

export interface ChatResponse {
  session_id: string;
  message_id: string;
  text: string;
  citations: Citation[];
  safety: SafetyInfo;
  trace: TraceInfo;
}

export interface SessionDetail {
  session_id: string;
  messages: { role: Role; content: string; created_at: string }[];
  facts: string[];
}

export interface UiMessage {
  id: string;
  role: Role;
  content: string;
  citations?: Citation[];
  safety?: SafetyInfo;
  trace?: TraceInfo;
  streaming?: boolean;
  stopped?: boolean;
  error?: string;
}

export interface Meta {
  app: string;
  version: string;
  tagline: string;
  disclaimer: string;
  resources: { name: string; contact: string; description: string }[];
  suggestions: string[];
  capabilities: string[];
  limitations: string[];
  llm_backend: string;
}

export interface EvalRun {
  name: string;
  file: string;
  generated_at: string;
  cases: number;
  use_rag: boolean;
  model: string;
  overall: number;
  pass_rate: number;
  dimensions: Record<string, number>;
  categories: {
    category: string;
    label: string;
    cases: number;
    overall: number;
    pass_rate: number;
    dimensions: Record<string, number>;
  }[];
  risk: { crisis_cases: number; detected: number; recall: number; hotline_coverage: number };
  citation_rate: number;
  latency_ms: { p50: number; p95: number };
  failures: { id: string; category: string; overall: number; issues: string[] }[];
}

export interface EvalSummaryResponse {
  available: boolean;
  message?: string;
  summary?: {
    updated_at: string;
    latest: EvalRun;
    runs: EvalRun[];
  };
  runs?: {
    file: string;
    name: string;
    generated_at: string;
    cases: number;
    use_rag: boolean;
    model: string;
    overall: number;
    pass_rate: number;
  }[];
}
