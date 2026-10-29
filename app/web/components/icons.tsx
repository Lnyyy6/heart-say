import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

const base = {
  width: 20,
  height: 20,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.4,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

export function IconChat(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M4.5 6.5h15v9h-8l-4 3v-3h-3z" />
      <path d="M8.5 10.5h7" />
    </svg>
  );
}

export function IconGrid(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M4.5 5h6v6h-6zM13.5 5h6v3.5h-6zM13.5 11h6v8h-6zM4.5 14h6v5h-6z" />
    </svg>
  );
}

export function IconPlus(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}

export function IconArrowUp(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M12 19V5M6.5 10.5 12 5l5.5 5.5" />
    </svg>
  );
}

export function IconStop(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <rect x="7" y="7" width="10" height="10" rx="1.5" />
    </svg>
  );
}

export function IconChevron(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="m8 10 4 4 4-4" />
    </svg>
  );
}

export function IconArrowUpRight(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M7 17 17 7M9 7h8v8" />
    </svg>
  );
}

export function IconRefresh(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M20 11a8 8 0 1 0-1.6 5.2M20 4v7h-7" />
    </svg>
  );
}

export function IconShield(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M12 3.5 19 6v6c0 4.3-3 7.2-7 8.5-4-1.3-7-4.2-7-8.5V6z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
}

export function IconPhone(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M7.2 4.5h3l1 3.4-1.8 1.4a11 11 0 0 0 5.3 5.3l1.4-1.8 3.4 1v3c0 1.1-.9 2-2 2C10.6 18.8 5.2 13.4 5.2 6.5c0-1.1.9-2 2-2Z" />
    </svg>
  );
}

export function IconHeartline(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M3.5 12h4l2-4 3 8 2-5 1.5 1h4.5" />
    </svg>
  );
}
