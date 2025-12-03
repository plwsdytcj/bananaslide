import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  // 从项目根目录读取 .env 文件
  envDir: path.resolve(__dirname, '..'),
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    host: true, // 监听所有地址
    watch: {
      usePolling: true, // WSL 环境下需要启用轮询
    },
    hmr: {
      overlay: true, // 显示错误覆盖层
    },
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
})

