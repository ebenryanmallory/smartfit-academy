import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  base: '/',
  server: {
    port: 5175, // Non-standard port; change if needed
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    cssMinify: true,
    rollupOptions: {
      output: {
        assetFileNames: 'assets/[name].[hash][extname]'
      }
    }
  },
  css: {
    postcss: './postcss.config.js',
    devSourcemap: true
  },
  plugins: [
    react(), 
    tailwindcss()
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
})
