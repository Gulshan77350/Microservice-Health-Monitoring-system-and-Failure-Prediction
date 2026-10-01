import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Dev-server proxy mirroring dashboard/nginx.conf.template, for `npm run dev`
// against backends on localhost. The X-API-Key is added here (server side),
// never shipped to the browser. Set API_KEY in the shell or in dashboard/.env.local.
export default defineConfig(({ mode }) => {
  const env = { ...loadEnv(mode, process.cwd(), ""), ...process.env };
  const keyHeader = { "X-API-Key": env.API_KEY ?? "local-demo-key" };
  const strip = (prefix) => (path) => path.replace(prefix, "");

  return {
    plugins: [react()],
    // recharts dominates the bundle (~190 kB gzipped); a single chunk is fine for this dashboard.
    build: { chunkSizeWarningLimit: 700 },
    server: {
      proxy: {
        "^/api/(model-info|feature-importance|drift|alert-config|alert)": {
          target: "http://localhost:8004",
          changeOrigin: true,
          rewrite: strip(/^\/api/),
          headers: keyHeader,
        },
        "/api/payment": { target: "http://localhost:8001", rewrite: strip(/^\/api\/payment/), headers: keyHeader },
        "/api/order": { target: "http://localhost:8002", rewrite: strip(/^\/api\/order/), headers: keyHeader },
        "/api/notification": {
          target: "http://localhost:8003",
          rewrite: strip(/^\/api\/notification/),
          headers: keyHeader,
        },
        "/collector": { target: "http://localhost:8005", rewrite: strip(/^\/collector/) },
      },
    },
  };
});
