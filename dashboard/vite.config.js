import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";

// GitHub Pages serves this repository's site at /<repo name>/, so every asset
// path must start there. Change it if the repository is renamed.
const base = "/DSA4262-Foursight/";

// The Explore page reads the full site export (analysis/sgnex/export_site_shortlist.py,
// ~90 MB). It is not committed (report/task2_plan.md, 5.3): the deploy workflow puts it in
// public/data/sites/ from a GitHub Release. Locally, `npm run dev` and `npm run preview`
// serve it straight from data/dashboard/ instead, so nothing needs copying.
// SITE_DATA=<folder> points them elsewhere.
function siteData() {
  const dir = path.resolve(process.env.SITE_DATA || path.join(__dirname, "..", "data", "dashboard"));
  const prefix = `${base}data/sites/`;
  const serve = (req, res, next) => {
    if (!req.url?.startsWith(prefix)) return next();
    const rel = decodeURIComponent(req.url.slice(prefix.length).split("?")[0]);
    const file = path.resolve(dir, rel);
    // Only files inside the export folder: never let "../" reach the course data beside it.
    const inside = path.relative(dir, file);
    if (!inside || inside.startsWith("..") || path.isAbsolute(inside)) return next();
    if (!fs.existsSync(file) || !fs.statSync(file).isFile()) return next();
    // Served as plain bytes: the page unpacks .gz itself, exactly as on GitHub Pages.
    res.setHeader("Content-Type", file.endsWith(".json") ? "application/json" : "application/octet-stream");
    fs.createReadStream(file).pipe(res);
  };
  return {
    name: "site-data",
    configureServer(server) { server.middlewares.use(serve); },
    configurePreviewServer(server) { server.middlewares.use(serve); },
  };
}

export default defineConfig({
  plugins: [react(), siteData()],
  base,
});
