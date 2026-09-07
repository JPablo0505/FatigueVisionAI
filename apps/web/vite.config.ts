import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      // /ws proxy REMOVED (S3-02): el frontend conecta directamente al
      // backend via VITE_WS_URL o ws://localhost:8000/ws, evitando el
      // ECONNABORTED del proxy de Vite por alto volumen de frames.
    },
  },
})