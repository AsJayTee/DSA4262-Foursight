import { useEffect, useState } from "react";

// Phones get simpler versions of the large diagrams (the 3D strand, the read tables, the
// network, the site graph); they do not work at that width. One breakpoint for all of them.
export const NARROW = "(max-width: 760px)";

export function useNarrow(query = NARROW) {
  const [match, setMatch] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const m = window.matchMedia(query);
    const on = () => setMatch(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, [query]);
  return match;
}
