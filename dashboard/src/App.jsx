import { useEffect, useState } from "react";
import Home from "./pages/Home.jsx";
import HowItWorks from "./pages/HowItWorks.jsx";
import Explore from "./pages/Explore.jsx";
import ModelStats from "./pages/ModelStats.jsx";
import { useRoute, Link } from "./lib/route.jsx";

const PAGES = [
  { path: "/", label: "Home", short: "Home", component: Home },
  { path: "/explore", label: "Explore sites", short: "Explore", component: Explore },
  { path: "/how-it-works", label: "How it works", short: "Method", component: HowItWorks },
  { path: "/model-stats", label: "Model stats", short: "Stats", component: ModelStats },
];

// Light or dark: the system's choice until the reader picks one with the toggle.
function useTheme() {
  const system = () => (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("fs-theme") || null;
    } catch {
      return null;
    }
  });
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
    try {
      if (theme) localStorage.setItem("fs-theme", theme);
    } catch {
      /* storage blocked: the choice lasts for this visit only */
    }
  }, [theme]);
  const current = theme || system();
  return [current, () => setTheme(current === "dark" ? "light" : "dark")];
}

function ThemeIcon({ dark }) {
  return dark ? (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
      <circle cx="12" cy="12" r="4.5" />
      {[0, 45, 90, 135, 180, 225, 270, 315].map((a) => (
        <line key={a} x1="12" y1="2.5" x2="12" y2="5" transform={`rotate(${a} 12 12)`} />
      ))}
    </svg>
  ) : (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round">
      <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5Z" />
    </svg>
  );
}

export default function App() {
  const route = useRoute();
  const [theme, toggleTheme] = useTheme();
  const page = PAGES.find((p) => p.path === route.path) || PAGES[0];
  const Page = page.component;

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <Link to="/" className="brand" aria-label="Foursight m6A, home">
            <span className="brand-mark" aria-hidden="true" />
            Foursight <span className="brand-sub">m6A</span>
          </Link>
          <div className="topbar-right">
            <nav aria-label="Main">
              {PAGES.map((p) => (
                <Link key={p.path} to={p.path} className={p === page ? "nav-link active" : "nav-link"}>
                  <span className="nav-full">{p.label}</span><span className="nav-short" aria-hidden="true">{p.short}</span>
                </Link>
              ))}
            </nav>
            <button className="theme-toggle" onClick={toggleTheme}
              aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}>
              <ThemeIcon dark={theme === "dark"} />
            </button>
          </div>
        </div>
      </header>

      <main id="main">
        <Page route={route} />
      </main>

      <footer className="footer">
        <div className="footer-inner">
          Team Foursight, DSA4262 (NUS / Genome Institute of Singapore), 2026. Predictions on public{" "}
          <a href="https://github.com/GoekeLab/sg-nex-data">SG-NEx</a> data.{" "}
          <a href="https://github.com/AsJayTee/DSA4262-Foursight">Code on GitHub</a>.
        </div>
      </footer>
    </>
  );
}
