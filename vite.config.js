import { defineConfig } from 'vite'
import wasm from 'vite-plugin-wasm'

const repositoryName = process.env.GITHUB_REPOSITORY?.split('/')[1]
const isUserSite = repositoryName?.endsWith('.github.io')
const githubPagesBase =
  repositoryName && !isUserSite ? `/${repositoryName}/` : '/'

export default defineConfig({
  base: process.env.GITHUB_ACTIONS ? githubPagesBase : '/',
  plugins: [wasm()],
  // Pre-bundling rewrites Rapier's `import ... from "*.wasm"` before
  // vite-plugin-wasm can handle it, which silently breaks the worker.
  optimizeDeps: { exclude: ['@dimforge/rapier2d'] },
  // Rapier is imported by the worker, so the worker build needs the plugin too.
  worker: {
    format: 'es',
    plugins: () => [wasm()],
  },
})