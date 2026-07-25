import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base: "./" keeps asset paths relative so the built deck works from any static host
export default defineConfig({
  plugins: [react()],
  base: "./",
});
