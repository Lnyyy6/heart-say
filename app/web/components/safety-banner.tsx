import type { SafetyInfo } from "@/lib/types";
import { IconShield } from "./icons";

export function SafetyBanner({ safety }: { safety: SafetyInfo }) {
  if (safety.level === "none" || safety.level === "low") return null;
  const urgent = safety.level === "high" || safety.level === "critical";

  return (
    <aside className={`safety-banner ${urgent ? "is-urgent" : ""}`} aria-label="安全提示">
      <div className="safety-icon">
        <IconShield />
      </div>
      <div className="safety-copy">
        <p className="eyebrow">{urgent ? "SAFETY FIRST" : "BOUNDARIES"}</p>
        <h3>{urgent ? "先把安全放在第一位" : "有些边界需要说清楚"}</h3>
        <p>
          {urgent
            ? "这段对话里出现了需要认真对待的风险信号。如果你或身边的人正处在危险中，请先联系现实中的支持，而不是独自等待。"
            : "这类问题涉及诊断、用药或隐私。我可以陪你梳理，但不能替代专业评估，也不会越过这些边界。"}
        </p>
        {safety.resources.length > 0 && (
          <div className="safety-resources">
            {safety.resources.map((resource) => {
              const phone = resource.contact.match(/1\d{2,4}/)?.[0];
              return (
                <span className="resource-chip" key={resource.name}>
                  <span className="resource-name">{resource.name}</span>
                  {phone ? (
                    <a href={`tel:${phone}`} className="resource-contact">
                      {resource.contact}
                    </a>
                  ) : (
                    <span className="resource-contact">{resource.contact}</span>
                  )}
                </span>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
}
