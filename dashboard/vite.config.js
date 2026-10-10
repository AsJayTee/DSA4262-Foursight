import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages serves this repository's site at /<repo name>/, so every asset
// path must start there. Change it if the repository is renamed.
export default defineConfig({
  plugins: [react()],
  base: "/DSA4262-Foursight/",
});
