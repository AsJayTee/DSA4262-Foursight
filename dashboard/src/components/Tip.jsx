import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

// A hover target styled like a small button. Resting on it for a moment opens a short
// popup beside the pointer (or beside the element, for keyboard focus). Rendered at the
// end of <body>, so scrolling tables and cards cannot clip it.
const DELAY = 450;

// Only one popup on screen at a time. The open Tip registers how to close itself; any popup
// that opens (another Tip, or a chart's hover popup) closes it first.
let closeOpenTip = null;
function claimScreen(close) {
  if (closeOpenTip && closeOpenTip !== close) closeOpenTip();
  closeOpenTip = close;
}
function releaseScreen(close) {
  if (closeOpenTip === close) closeOpenTip = null;
}

// as="button": a real button (e.g. a mode toggle) that also explains itself on a dwell;
// extra props (onClick, aria-pressed, ...) pass through to it.
// clickToggle: a click or tap also opens and closes the popup (the "i" buttons), and it then
// stays open until the next click elsewhere. Without it a click closes the popup, which suits
// targets whose click does something else.
export function Tip({ text, children, className = "", label, as = "span", clickToggle = false, ...rest }) {
  const [at, setAt] = useState(null);         // {x, y} where the popup opens
  const timer = useRef(null);
  const pos = useRef({ x: 0, y: 0 });
  const ref = useRef(null);
  const pinned = useRef(false);               // opened by a click: ignore pointer leave
  const close = () => { clearTimeout(timer.current); pinned.current = false; setAt(null); };
  // A stable handle for the shared register: `close` itself is recreated every render.
  const closeRef = useRef(close);
  closeRef.current = close;
  const handle = useRef(() => closeRef.current()).current;
  const Tag = as;
  useEffect(() => () => { clearTimeout(timer.current); releaseScreen(handle); }, [handle]);
  useEffect(() => {
    if (!at) { releaseScreen(handle); return; }
    claimScreen(handle);
    const onKey = (e) => e.key === "Escape" && close();
    const onDown = (e) => { if (!ref.current?.contains(e.target)) close(); };
    window.addEventListener("keydown", onKey);
    window.addEventListener("scroll", close, { passive: true, once: true });
    window.addEventListener("pointerdown", onDown);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", close);
      window.removeEventListener("pointerdown", onDown);
    };
  }, [at]);
  const besideTarget = () => { const r = ref.current.getBoundingClientRect(); return { x: r.right, y: r.bottom }; };
  return (
    <>
      <Tag ref={ref} className={`tip-target ${className}`} aria-label={label}
        {...(as === "button" ? { type: "button" } : { tabIndex: "0", role: "button" })} {...rest}
        aria-describedby={at ? "tip-pop" : undefined} aria-expanded={clickToggle ? !!at : undefined}
        onPointerEnter={(e) => {
          // A touch has no hover; its tap is handled as a click below.
          if (e.pointerType === "touch") return;
          claimScreen(handle);                   // close another open popup now, not after the delay
          pos.current = { x: e.clientX, y: e.clientY };
          clearTimeout(timer.current);
          // A click-toggled popup opens beside its button whether hovered or clicked, so a click
          // on an open popup keeps it exactly where it is.
          timer.current = setTimeout(() => setAt(clickToggle ? besideTarget() : { ...pos.current }), DELAY);
        }}
        onPointerMove={(e) => { pos.current = { x: e.clientX, y: e.clientY }; }}
        onPointerLeave={(e) => { if (e.pointerType !== "touch" && !pinned.current) close(); }}
        onFocus={(e) => {
          // Keyboard focus only: a mouse click on a button should not open the popup.
          if (!e.currentTarget.matches(":focus-visible")) return;
          setAt(besideTarget());
        }}
        onBlur={() => { if (!pinned.current) close(); }}
        onKeyDown={(e) => {
          if (clickToggle && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            if (at) close(); else { pinned.current = true; setAt(besideTarget()); }
          }
          rest.onKeyDown?.(e);
        }}
        onClick={(e) => {
          e.stopPropagation();
          if (clickToggle) {
            if (pinned.current) { close(); return; }
            // Pin it open. If hovering already opened it, leave it in place rather than reopening.
            clearTimeout(timer.current);
            pinned.current = true;
            if (!at) setAt(besideTarget());
            return;
          }
          close();
          rest.onClick?.(e);
        }}>
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
// side="left" opens it to the left of the point instead of the right. vertical="slide" keeps a
// popup that sits beside its target level with the point, moving it up only as far as needed to
// fit the window; the default "flip" puts it wholly above the point, which suits small tooltips
// but sends a tall one far up the page.
export function Popup({ at, children, id, side = "right", vertical = "flip" }) {
  const ref = useRef(null);
  const [style, setStyle] = useState({ left: at.x + 14, top: at.y + 16, visibility: "hidden" });
  // A chart's hover popup takes the screen from any open Tip (a Tip's own popup has id tip-pop).
  useEffect(() => { if (id !== "tip-pop") claimScreen(null); }, [id]);
  // Keep it on screen: flip left, and flip or slide up, when it would overflow.
  useEffect(() => {
    const r = ref.current.getBoundingClientRect();
    const fitsRight = at.x + 14 + r.width <= window.innerWidth - 8;
    const left = side === "left" || !fitsRight ? at.x - 14 - r.width : at.x + 14;
    const below = at.y + 16 + r.height <= window.innerHeight - 8;
    const top = below ? at.y + 16 : vertical === "slide" ? window.innerHeight - 8 - r.height : at.y - 12 - r.height;
    setStyle({ left: Math.max(8, left), top: Math.max(8, top) });
  }, [at, side, vertical]);
  return createPortal(<div id={id} role="tooltip" ref={ref} className="tip-pop" style={style}>{children}</div>, document.body);
}

// The small "i" next to a column heading.
export function InfoTip({ text, label = "More information" }) {
  return <Tip text={text} className="info-btn" label={label} clickToggle>i</Tip>;
}
