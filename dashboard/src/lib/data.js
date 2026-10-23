import { useEffect, useState } from "react";

// Every data file is written by a script under analysis/ (dashboard/README.md); the
// base path is the GitHub Pages sub-path from vite.config.js.
const base = `${import.meta.env.BASE_URL}data/`;

export function useJson(name) {
  const [state, setState] = useState({ data: null, error: null });
  useEffect(() => {
    const controller = new AbortController();
    fetch(base + name, { signal: controller.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${name}: HTTP ${r.status}`))))
      .then((data) => setState({ data, error: null }))
      .catch((e) => {
        if (e.name !== "AbortError")
          setState({ data: null, error: `Could not load ${name}. Regenerate it with the script named in dashboard/README.md.` });
      });
    return () => controller.abort();
  }, [name]);
  return state;
}

// A within-file rank (0-1, higher = higher priority) as people read it: "top 2.4%".
export function topPercent(rank) {
  const top = (1 - rank) * 100;
  if (top < 0.1) return "top 0.1%";
  if (top < 9.95) return `top ${top.toFixed(1)}%`;
  return `top ${Math.round(top)}%`;
}

export const pct = (x, digits = 0) => `${(x * 100).toFixed(digits)}%`;
