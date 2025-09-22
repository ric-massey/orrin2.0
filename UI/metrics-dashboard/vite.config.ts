import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // your browser hits http://localhost:5173/metrics → Vite proxies to Python on 9100
      "/metrics": "http://localhost:9100"
    }
  }
});
