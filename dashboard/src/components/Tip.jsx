import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

// A hover target styled like a small button. Resting on it for a moment opens a short
// popup beside the pointer (or beside the element, for keyboard focus). Rendered at the
// end of <body>, so scrolling tables and cards cannot clip it.
const DELAY = 450;

// as="button": a real button (e.g. a mode toggle) that also explains itself on a dwell;
// extra props (onClick, aria-pressed, ...) pass through to it.
export function Tip({ text, children, className = "", label, as = "span", ...rest }) {
  const [at, setAt] = useState(null);         // {x, y} where the popup opens
  const timer = useRef(null);
  const pos = useRef({ x: 0, y: 0 });
  const ref = useRef(null);
  const close = () => { clearTimeout(timer.current); setAt(null); };
  const Tag = as;
  useEffect(() => () => clearTimeout(timer.current), []);
  useEffect(() => {
    if (!at) return;
    const onKey = (e) => e.key === "Escape" && close();
    window.addEventListener("keydown", onKey);
    window.addEventListener("scroll", close, { passive: true, once: true });
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("scroll", close); };
  }, [at]);
  return (
    <>
      <Tag ref={ref} className={`tip-target ${className}`} aria-label={label}
        {...(as === "button" ? { type: "button" } : { tabIndex: "0", role: "button" })} {...rest}
        aria-describedby={at ? "tip-pop" : undefined}
        onPointerEnter={(e) => {
          pos.current = { x: e.clientX, y: e.clientY };
          clearTimeout(timer.current);
          timer.current = setTimeout(() => setAt({ ...pos.current }), DELAY);
        }}
        onPointerMove={(e) => { pos.current = { x: e.clientX, y: e.clientY }; }}
        onPointerLeave={close}
        onFocus={(e) => {
          // Keyboard focus only: a mouse click on a button should not open the popup.
          if (!e.currentTarget.matches(":focus-visible")) return;
          const r = ref.current.getBoundingClientRect();
          setAt({ x: r.right, y: r.bottom });
        }}
        onBlur={close}
        onClick={(e) => { e.stopPropagation(); close(); rest.onClick?.(e); }}>
        {children}
      </Tag>
      {at && <TipPopup at={at} text={text} />}
    </>
  );
}

function TipPopup({ at, text }) {
  return <Popup at={at} id="tip-pop">{text}</Popup>;
}

// The same popup, opened directly at a screen point (charts use it on hover, no delay).
export function Popup({ at, children, id }) {
  const ref = useRef(null);
  const [style, setStyle] = useState({ left: at.x + 14, top: at.y + 16, visibility: "hidden" });
  // Keep it on screen: flip left or up when it would overflow.
  useEffect(() => {
    const r = ref.current.getBoundingClientRect();
    const left = at.x + 14 + r.width > window.innerWidth - 8 ? at.x - 14 - r.width : at.x + 14;
    const top = at.y + 16 + r.height > window.innerHeight - 8 ? at.y - 12 - r.height : at.y + 16;
    setStyle({ left: Math.max(8, left), top: Math.max(8, top) });
  }, [at]);
  return createPortal(<div id={id} role="tooltip" ref={ref} className="tip-pop" style={style}>{children}</div>, document.body);
}

// The small "i" next to a column heading.
export function InfoTip({ text, label = "More information" }) {
  return <Tip text={text} className="info-btn" label={label}>i</Tip>;
}
