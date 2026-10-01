import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Prediction API  →  http://localhost:8004
      '/api/predict': {
        target: 'http://localhost:8004',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api\/predict/, '/predict'),
      },
      // Model info & feature importance  →  http://localhost:8004
      '/api/model-info': {
        target: 'http://localhost:8004',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api\/model-info/, '/model-info'),
      },
      '/api/feature-importance': {
        target: 'http://localhost:8004',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api\/feature-importance/, '/feature-importance'),
      },
      '/api/drift': {
        target: 'http://localhost:8004',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api\/drift/, '/drift'),
      },
      '/api/alert-config': {
        target: 'http://localhost:8004',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api\/alert-config/, '/alert-config'),
      },
      // Payment service  →  http://localhost:8001
      '/api/payment': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api\/payment/, ''),
      },
      // Order service  →  http://localhost:8002
      '/api/order': {
        target: 'http://localhost:8002',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api\/order/, ''),
      },
      // Notification service  →  http://localhost:8003
      '/api/notification': {
        target: 'http://localhost:8003',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api\/notification/, ''),
      },
      // Collector API  →  http://localhost:8005
      '/collector': {
        target: 'http://localhost:8005',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/collector/, ''),
      },
    },
  },
})
