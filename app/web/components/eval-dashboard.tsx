"use client";

import { useEffect, useMemo, useState } from "react";
import { getEvalSummary } from "@/lib/api";
import type { EvalRun, EvalSummaryResponse } from "@/lib/types";
import { IconRefresh } from "./icons";
import { Rail } from "./rail";

const DIMENSIONS: { key: string; label: string; en: string }[] = [
  { key: "empathy", label: "共情", en: "EMPATHY" },
  { key: "boundary", label: "边界", en: "BOUNDARY" },
  { key: "safety", label: "安全", en: "SAFETY" },
  { key: "hallucination", label: "事实 / 引用", en: "GROUNDING" },
  { key: "persona", label: "人设一致", en: "PERSONA" },
  { key: "preachiness", label: "低说教感", en: "TONE" },
];

function pct(value: number) {
  return `${Math.round(value * 100)}%`;
}

function delta(value: number, digits = 2) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

export function EvalDashboard() {
  const [data, setData] = useState<EvalSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedName, setSelectedName] = useState("");

  const load = () => {
    setLoading(true);
    setError("");
    getEvalSummary()
      .then((payload) => {
        setData(payload);
        if (payload.summary?.latest) setSelectedName(payload.summary.latest.name);
      })
      .catch((reason: Error) => setError(reason.message || "无法读取评测报告"))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  const runs: EvalRun[] = useMemo(() => {
    if (!data?.summary) return [];
    const all = data.summary.runs?.length ? data.summary.runs : [data.summary.latest];
    const meaningful = all.filter((item) => !["smoke", "test"].includes(item.name));
    return meaningful.length ? meaningful : all;
  }, [data]);

  const run = useMemo(
    () => runs.find((item) => item.name === selectedName) ?? runs[runs.length - 1],
    [runs, selectedName],
  );

  const baseline =
    runs.find((item) => item.name === "baseline") ?? runs.find((item) => !item.use_rag);
  const rag = runs.find((item) => item.name === "rag") ?? runs.find((item) => item.use_rag);

  const refresh = () => load();

  return (
    <div className="shell">
      <Rail active="eval" />
      <div className="stage">
        <main className="main eval-main">
          <header className="main-head">
            <div>
              <p className="eyebrow">EVALUATION · VERIFIABLE CARE</p>
              <h2>评测</h2>
            </div>
            <div className="head-meta">
              <button type="button" className="icon-button" onClick={refresh} aria-label="刷新评测">
                <IconRefresh />
              </button>
              <span className="head-hint">自动 judge · 可复现</span>
            </div>
          </header>

          <div className="eval-scroll">
            <div className="eval-inner">
              <section className="eval-hero">
                <p className="eyebrow hero-eyebrow">
                  <span className="eyebrow-line" />
                  {run ? `${run.cases} CASES · ${run.use_rag ? "RAG ON" : "RAG OFF"}` : "NO RUN YET"}
                </p>
                <h1>
                  把共情，
                  <br />
                  <span>变成可验证的指标。</span>
                </h1>
                <p className="hero-lede">
                  共情、边界、安全、事实、人设与说教感，被拆成可以回归的维度。
                  <br />
                  分数不是终点，是下一次改动前的参照。
                </p>
              </section>

              {loading && <div className="session-loading">正在读取评测产物…</div>}

              {!loading && error && (
                <div className="eval-empty">
                  <p className="eyebrow">CONNECTION</p>
                  <h3>暂时读不到评测报告</h3>
                  <p>{error}</p>
                  <button type="button" className="outline-button" onClick={refresh}>
                    重试
                  </button>
                </div>
              )}

              {!loading && !error && !data?.available && (
                <div className="eval-empty">
                  <p className="eyebrow">EMPTY</p>
                  <h3>还没有评测产物</h3>
                  <p>在项目根目录运行下面命令，生成 baseline 与 RAG 两轮评测，然后刷新此页。</p>
                  <pre>
                    <code>{`make eval\n# 或\nPYTHONPATH=src .venv/bin/python -m heart_say.evaluation.runner --name rag`}</code>
                  </pre>
                </div>
              )}

              {!loading && !error && data?.available && run && (
                <>
                  <div className="run-switcher" role="tablist" aria-label="评测运行">
                    {runs.map((item) => (
                      <button
                        type="button"
                        role="tab"
                        aria-selected={item.name === run.name}
                        className={`run-chip ${item.name === run.name ? "is-active" : ""}`}
                        key={item.name}
                        onClick={() => setSelectedName(item.name)}
                      >
                        <span>{item.name}</span>
                        <em>{item.overall}</em>
                      </button>
                    ))}
                  </div>

                  <div className="metric-grid">
                    <div className="metric-card">
                      <p className="metric-label">总均分</p>
                      <p className="metric-value">
                        {run.overall}
                        <span>/ 5</span>
                      </p>
                      <p className="metric-note">{run.cases} 条对话用例</p>
                    </div>
                    <div className="metric-card">
                      <p className="metric-label">通过率</p>
                      <p className="metric-value">{pct(run.pass_rate)}</p>
                      <p className="metric-note">阈值 3.5 / 5</p>
                    </div>
                    <div className="metric-card">
                      <p className="metric-label">危机识别</p>
                      <p className="metric-value">{pct(run.risk.recall)}</p>
                      <p className="metric-note">
                        热线覆盖 {pct(run.risk.hotline_coverage)} · {run.risk.crisis_cases} 例
                      </p>
                    </div>
                    <div className="metric-card">
                      <p className="metric-label">引用覆盖</p>
                      <p className="metric-value">{pct(run.citation_rate)}</p>
                      <p className="metric-note">P95 {run.latency_ms.p95} ms</p>
                    </div>
                  </div>

                  {baseline && rag && baseline.name !== rag.name && (
                    <section className="compare-card">
                      <div>
                        <p className="eyebrow">BASELINE → RAG</p>
                        <h3>检索让事实更稳，风格没有被牺牲。</h3>
                      </div>
                      <div className="compare-stats">
                        <span>
                          <em>{delta(rag.overall - baseline.overall)}</em>
                          总均分
                        </span>
                        <span>
                          <em>{delta((rag.citation_rate - baseline.citation_rate) * 100, 0)} pp</em>
                          引用覆盖
                        </span>
                        <span>
                          <em>{delta(rag.latency_ms.p95 - baseline.latency_ms.p95, 1)}</em>
                          P95 ms
                        </span>
                      </div>
                    </section>
                  )}

                  <section className="eval-section">
                    <div className="section-head">
                      <p className="eyebrow">DIMENSIONS</p>
                      <h3>六个维度</h3>
                    </div>
                    <div className="dimension-list">
                      {DIMENSIONS.map((dimension) => {
                        const value = run.dimensions[dimension.key] ?? 0;
                        return (
                          <div className="dimension-row" key={dimension.key}>
                            <div className="dimension-label">
                              <span className="dimension-en">{dimension.en}</span>
                              <span>{dimension.label}</span>
                            </div>
                            <div className="dimension-track">
                              <span style={{ width: `${(value / 5) * 100}%` }} />
                            </div>
                            <span className="dimension-value">{value.toFixed(2)}</span>
                          </div>
                        );
                      })}
                    </div>
                  </section>

                  <section className="eval-section">
                    <div className="section-head">
                      <p className="eyebrow">BREAKDOWN</p>
                      <h3>分类表现</h3>
                    </div>
                    <div className="category-table">
                      <div className="category-row category-header">
                        <span>分类</span>
                        <span>用例</span>
                        <span>均分</span>
                        <span>通过率</span>
                      </div>
                      {run.categories.map((category) => (
                        <div className="category-row" key={category.category}>
                          <span>{category.label}</span>
                          <span>{category.cases}</span>
                          <span>{category.overall.toFixed(2)}</span>
                          <span className="category-pass">
                            <span className="mini-track">
                              <span style={{ width: pct(category.pass_rate) }} />
                            </span>
                            {pct(category.pass_rate)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </section>

                  <section className="eval-section">
                    <div className="section-head">
                      <p className="eyebrow">FAILURES</p>
                      <h3>还需要看的样本</h3>
                    </div>
                    {run.failures.length ? (
                      <div className="failure-list">
                        {run.failures.map((failure) => (
                          <article className="failure-card" key={failure.id}>
                            <div className="failure-top">
                              <span className="failure-id">{failure.id}</span>
                              <span className="failure-category">{failure.category}</span>
                              <span className="failure-score">{failure.overall}</span>
                            </div>
                            <ul>
                              {failure.issues.map((issue) => (
                                <li key={issue}>{issue}</li>
                              ))}
                            </ul>
                          </article>
                        ))}
                      </div>
                    ) : (
                      <p className="aside-muted">当前运行没有失败样本。仍需人工抽检确认。</p>
                    )}
                  </section>

                  <p className="eval-footnote">
                    Judge：{run.model} · 报告生成于 {new Date(run.generated_at).toLocaleString("zh-CN")}。
                    自动评分是可复现的回归信号，不等同于人工评审。
                  </p>
                </>
              )}
            </div>
          </div>
        </main>

        <aside className="aside">
          <section className="breath-card eval-aside-card">
            <div>
              <p className="eyebrow">WHY EVAL</p>
              <h3>温柔也需要证据。</h3>
              <p>
                把“听起来舒服”拆成可检查的行为：先共情、守边界、危机转介、引用克制、不说教。
              </p>
            </div>
          </section>

          <section className="aside-card">
            <p className="eyebrow">METHOD</p>
            <ul className="principle-list">
              <li>
                <span>01</span>规则 judge 做版本回归，LLM judge 可选
              </li>
              <li>
                <span>02</span>危机、边界、事实类用例单独看安全分
              </li>
              <li>
                <span>03</span>发布前按 10-20% 人工抽检校准
              </li>
            </ul>
          </section>

          <section className="aside-card">
            <p className="eyebrow">REPRODUCE</p>
            <pre className="aside-code">
              <code>{`make eval\nmake redteam\nmake test`}</code>
            </pre>
            <p className="aside-muted">
              报告写入 reports/，包含 baseline、RAG 对比、红队结果与性能数据。
            </p>
          </section>

          <section className="aside-card crisis-card">
            <p className="eyebrow">NOTE</p>
            <p>
              分数不能替代真实关怀。若你或他人正处在危险中，请联系 12356 或 120/110。
            </p>
          </section>

          <p className="aside-disclaimer">
            自动评分是研究/工程指标，不构成医疗结论。
          </p>
        </aside>
      </div>
    </div>
  );
}
