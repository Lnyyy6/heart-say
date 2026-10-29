"use client";

import { useEffect, useRef, useState } from "react";
import { getMeta, getSession, streamChat } from "@/lib/api";
import type { Meta, SafetyInfo, UiMessage } from "@/lib/types";
import { Composer } from "./composer";
import { IconArrowUpRight, IconShield } from "./icons";
import { Message } from "./message";
import { Rail } from "./rail";
import { SafetyBanner } from "./safety-banner";

const SESSION_KEY = "heart_say_session";

const FALLBACK_SUGGESTIONS = [
  "最近孩子总说不想上学，我该怎么开口？",
  "一考试就睡不着，总怕考砸。",
  "孩子把自己关在房间，不愿意跟我说话。",
  "我好像总是控制不住吼孩子，事后又很自责。",
];

function makeId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `hs-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function humanizeFact(fact: string) {
  if (fact.startsWith("身份：")) return `你是${fact.slice(3)}`;
  if (fact.startsWith("反复提到：")) return `${fact.slice(5)}被反复提起`;
  if (fact.startsWith("年级：")) return fact.slice(3);
  return fact;
}

export function Chat() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [facts, setFacts] = useState<string[]>([]);
  const [sessionId, setSessionId] = useState("");
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [activeSafety, setActiveSafety] = useState<SafetyInfo | null>(null);
  const [loadingSession, setLoadingSession] = useState(true);

  const threadRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);
  const abortRef = useRef<AbortController | null>(null);
  const busyRef = useRef(false);
  const sessionRef = useRef("");

  const suggestions = (meta?.suggestions?.length ? meta.suggestions : FALLBACK_SUGGESTIONS).slice(0, 4);

  useEffect(() => {
    getMeta().then(setMeta).catch(() => undefined);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const existing = window.localStorage.getItem(SESSION_KEY);
    if (!existing) {
      const id = makeId();
      window.localStorage.setItem(SESSION_KEY, id);
      sessionRef.current = id;
      setSessionId(id);
      setLoadingSession(false);
      return;
    }
    sessionRef.current = existing;
    setSessionId(existing);
    getSession(existing)
      .then((session) => {
        if (cancelled) return;
        setFacts(session.facts);
        setMessages(
          session.messages.map((message) => ({
            id: makeId(),
            role: message.role,
            content: message.content,
          })),
        );
      })
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled) setLoadingSession(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const node = threadRef.current;
    if (!node || !stickRef.current) return;
    node.scrollTop = node.scrollHeight;
  }, [messages, activeSafety]);

  const refreshFacts = (sid: string) => {
    getSession(sid)
      .then((session) => setFacts(session.facts))
      .catch(() => undefined);
  };

  const handleNew = () => {
    abortRef.current?.abort();
    const id = makeId();
    window.localStorage.setItem(SESSION_KEY, id);
    sessionRef.current = id;
    setSessionId(id);
    setMessages([]);
    setFacts([]);
    setInput("");
    setActiveSafety(null);
    setStreaming(false);
    busyRef.current = false;
    stickRef.current = true;
  };

  const handleSend = (text?: string) => {
    const content = (text ?? input).trim();
    if (!content || busyRef.current) return;

    let sid = sessionRef.current;
    if (!sid) {
      sid = makeId();
      window.localStorage.setItem(SESSION_KEY, sid);
      sessionRef.current = sid;
      setSessionId(sid);
    }

    const userMessage: UiMessage = { id: makeId(), role: "user", content };
    const assistantId = makeId();
    const assistantMessage: UiMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      streaming: true,
    };

    setMessages((previous) => [...previous, userMessage, assistantMessage]);
    setInput("");
    setActiveSafety(null);
    setStreaming(true);
    busyRef.current = true;
    stickRef.current = true;

    const controller = new AbortController();
    abortRef.current = controller;
    let received = "";

    const patchAssistant = (patch: Partial<UiMessage>) => {
      setMessages((previous) =>
        previous.map((message) => (message.id === assistantId ? { ...message, ...patch } : message)),
      );
    };

    void streamChat(
      { message: content, session_id: sid, use_rag: true },
      {
        onMeta: (payload) => {
          setActiveSafety(payload.safety);
          patchAssistant({ safety: payload.safety, trace: payload.trace });
        },
        onDelta: (chunk) => {
          received += chunk;
          setMessages((previous) =>
            previous.map((message) =>
              message.id === assistantId
                ? { ...message, content: message.content + chunk }
                : message,
            ),
          );
        },
        onCitations: (citations) => patchAssistant({ citations }),
        onDone: (response) => {
          patchAssistant({
            content: response.text || received,
            citations: response.citations,
            safety: response.safety,
            trace: response.trace,
            streaming: false,
          });
          setActiveSafety(response.safety);
          refreshFacts(sid);
        },
        onError: (message) => {
          patchAssistant({ streaming: false, error: message });
        },
      },
      controller.signal,
    )
      .catch((error: Error) => {
        if (error.name === "AbortError") {
          patchAssistant({ streaming: false, stopped: true });
          return;
        }
        patchAssistant({
          streaming: false,
          error: error.message || "连接中断，请稍后重试。",
        });
      })
      .finally(() => {
        setStreaming(false);
        busyRef.current = false;
        abortRef.current = null;
        setMessages((previous) =>
          previous.map((message) =>
            message.id === assistantId && message.streaming
              ? { ...message, streaming: false, stopped: true }
              : message,
          ),
        );
      });
  };

  const handleStop = () => {
    abortRef.current?.abort();
  };

  const handleRetry = () => {
    const last = messages[messages.length - 1];
    if (!last || last.role !== "assistant") return;
    let userIndex = -1;
    for (let index = messages.length - 2; index >= 0; index -= 1) {
      if (messages[index].role === "user") {
        userIndex = index;
        break;
      }
    }
    if (userIndex < 0) return;
    const text = messages[userIndex].content;
    busyRef.current = false;
    setMessages((previous) => previous.slice(0, userIndex));
    handleSend(text);
  };

  const onThreadScroll = () => {
    const node = threadRef.current;
    if (!node) return;
    stickRef.current = node.scrollHeight - node.scrollTop - node.clientHeight < 140;
  };

  const showHero = !loadingSession && messages.length === 0;

  return (
    <div className="shell">
      <Rail active="chat" onNew={handleNew} />
      <div className="stage">
        <main className="main">
          <header className="main-head">
            <div>
              <p className="eyebrow">
                CONVERSATION · {sessionId ? sessionId.slice(0, 6).toUpperCase() : "NEW"}
              </p>
              <h2>对话</h2>
            </div>
            <div className="head-meta">
              <span className="head-status">
                <span className="status-dot" aria-hidden="true" />
                陪伴中
              </span>
              <span className="head-hint">上下文记忆已开启</span>
            </div>
          </header>

          {activeSafety && (
            <div className="safety-slot">
              <SafetyBanner safety={activeSafety} />
            </div>
          )}

          <div className="thread" ref={threadRef} onScroll={onThreadScroll}>
            <div className="thread-inner">
              {loadingSession && <div className="session-loading">正在接上上一段对话…</div>}

              {showHero && (
                <section className="hero">
                  <p className="eyebrow hero-eyebrow">
                    <span className="eyebrow-line" />
                    AN INVITATION, NOT A DIAGNOSIS
                  </p>
                  <h1>
                    先说感受，
                    <br />
                    <span>再找办法。</span>
                  </h1>
                  <p className="hero-lede">
                    不急着给答案。先听懂你，再一起找一个能走的小步。
                    <br />
                    这里没有评判，也不做诊断。
                  </p>
                  <div className="suggestions">
                    {suggestions.map((suggestion, index) => (
                      <button
                        type="button"
                        className="suggestion"
                        key={suggestion}
                        onClick={() => handleSend(suggestion)}
                        style={{ animationDelay: `${180 + index * 70}ms` }}
                      >
                        <span className="suggestion-index">0{index + 1}</span>
                        <span className="suggestion-text">{suggestion}</span>
                        <IconArrowUpRight className="suggestion-arrow" />
                      </button>
                    ))}
                  </div>
                  <p className="hero-note">
                    <span className="hero-note-line" />
                    不诊断 · 不评判 · 危机时优先安全
                  </p>
                </section>
              )}

              {messages.map((message, index) => (
                <Message
                  key={message.id}
                  message={message}
                  isLast={index === messages.length - 1}
                  onRetry={handleRetry}
                />
              ))}
              <div className="thread-end" aria-hidden="true" />
            </div>
          </div>

          <div className="composer-dock">
            <Composer
              value={input}
              onChange={setInput}
              onSend={() => handleSend()}
              onStop={handleStop}
              streaming={streaming}
            />
          </div>
        </main>

        <aside className="aside">
          <section className="breath-card">
            <div className="breath-visual" aria-hidden="true">
              <span />
              <span />
              <span />
              <i />
            </div>
            <div>
              <p className="eyebrow">PAUSE / 01</p>
              <h3>先慢一点。</h3>
              <p>不急着给建议。你可以从最难受的那个瞬间说起，我会在。</p>
            </div>
          </section>

          <section className="aside-card">
            <p className="eyebrow">HOW I LISTEN</p>
            <ul className="principle-list">
              <li>
                <span>01</span>先复述你的感受，不急着下结论
              </li>
              <li>
                <span>02</span>给具体的小步骤，不做长篇说教
              </li>
              <li>
                <span>03</span>不诊断、不荐药，危机时先转介
              </li>
            </ul>
          </section>

          <section className="aside-card">
            <p className="eyebrow">MEMORY</p>
            {facts.length ? (
              <>
                <h3 className="aside-title">我记得</h3>
                <ul className="fact-list">
                  {facts.slice(-4).map((fact) => (
                    <li key={fact}>{humanizeFact(fact)}</li>
                  ))}
                </ul>
              </>
            ) : (
              <p className="aside-muted">聊过几轮后，我会记住重要的背景，让下一次不必从头说起。</p>
            )}
          </section>

          <section className="aside-card crisis-card">
            <div className="crisis-head">
              <IconShield />
              <span>如果此刻有危险</span>
            </div>
            <p>不要独自等待。优先联系现实中的支持。</p>
            <div className="crisis-lines">
              <a href="tel:12356">
                <span>心理援助热线</span>
                <strong>12356</strong>
              </a>
              <div>
                <span>紧急救援</span>
                <strong>
                  <a href="tel:120">120</a>
                  <em>/</em>
                  <a href="tel:110">110</a>
                </strong>
              </div>
            </div>
          </section>

          <p className="aside-disclaimer">
            心语提供心理教育与情绪陪伴，不进行医疗诊断，也不替代心理咨询或精神科治疗。
          </p>
        </aside>
      </div>
    </div>
  );
}
