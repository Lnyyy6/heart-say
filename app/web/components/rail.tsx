"use client";

import Link from "next/link";
import { BrandMark } from "./brand";
import { IconChat, IconGrid, IconPlus } from "./icons";

export function Rail({
  active,
  onNew,
}: {
  active: "chat" | "eval";
  onNew?: () => void;
}) {
  const handleNew = () => {
    if (onNew) {
      onNew();
      return;
    }
    if (typeof window !== "undefined") {
      window.localStorage.removeItem("heart_say_session");
      window.location.href = "/";
    }
  };

  return (
    <aside className="rail" aria-label="主导航">
      <Link href="/" className="rail-brand" aria-label="心语首页">
        <BrandMark size={30} />
      </Link>
      <div className="rail-vertical" aria-hidden="true">
        HEART SAY
      </div>
      <nav className="rail-nav">
        <Link
          href="/"
          className={`rail-item ${active === "chat" ? "is-active" : ""}`}
          aria-current={active === "chat" ? "page" : undefined}
          data-label="对话"
        >
          <IconChat />
          <span>对话</span>
        </Link>
        <button type="button" className="rail-item" onClick={handleNew} data-label="新对话">
          <IconPlus />
          <span>新对话</span>
        </button>
        <Link
          href="/eval"
          className={`rail-item ${active === "eval" ? "is-active" : ""}`}
          aria-current={active === "eval" ? "page" : undefined}
          data-label="评测"
        >
          <IconGrid />
          <span>评测</span>
        </Link>
      </nav>
      <div className="rail-foot">
        <span className="status-dot" aria-hidden="true" />
        <span className="rail-foot-label">LOCAL</span>
      </div>
    </aside>
  );
}
