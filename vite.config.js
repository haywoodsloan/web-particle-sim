import { defineConfig } from 'vite'

const repositoryName = process.env.GITHUB_REPOSITORY?.split('/')[1]
const isUserSite = repositoryName?.endsWith('.github.io')
const githubPagesBase =
  repositoryName && !isUserSite ? `/${repositoryName}/` : '/'

// SharedArrayBuffer, and so the physics thread pool, only exists in a
// cross-origin isolated page. GitHub Pages cannot send these, so the service
// worker in public/ replays them there.
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'same-origin',
}

export default defineConfig({
  base: process.env.GITHUB_ACTIONS ? githubPagesBase : '/',
  worker: { format: 'es' },
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
})