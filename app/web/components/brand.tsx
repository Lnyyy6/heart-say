export function BrandMark({ size = 34 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 40 40"
      fill="none"
      aria-hidden="true"
      className="brand-mark"
    >
      <circle cx="20" cy="20" r="16.4" stroke="currentColor" strokeWidth="0.8" opacity="0.45" />
      <circle cx="20" cy="20" r="11.8" stroke="currentColor" strokeWidth="0.8" opacity="0.2" />
      <path
        d="M12.5 21.5c2.4-5.4 5.1-5.4 7.5 0s5.1 5.4 7.5 0"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
      />
      <circle cx="20" cy="20" r="1.2" fill="currentColor" />
    </svg>
  );
}

export function Wordmark() {
  return (
    <span className="wordmark">
      <span className="wordmark-cn">心语</span>
      <span className="wordmark-en">HEART SAY</span>
    </span>
  );
}
