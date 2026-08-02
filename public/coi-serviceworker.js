/**
 * Cross-origin isolation shim.
 *
 * SharedArrayBuffer needs COOP/COEP response headers and GitHub Pages cannot
 * send them, so this file doubles as its own service worker: the page registers
 * it, and it re-serves every response with the headers attached. The one reload
 * is unavoidable because a freshly registered worker does not control the page
 * that registered it.
 */
if (typeof window === 'undefined') {
  self.addEventListener('install', () => self.skipWaiting())
  self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()))

  self.addEventListener('fetch', (event) => {
    const { request } = event

    // Range requests served from cache cannot be rewritten, so leave them be.
    if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') {
      return
    }

    event.respondWith(
      fetch(request).then((response) => {
        // Opaque responses have an immutable header list.
        if (response.status === 0) {
          return response
        }

        const headers = new Headers(response.headers)

        headers.set('Cross-Origin-Embedder-Policy', 'require-corp')
        headers.set('Cross-Origin-Opener-Policy', 'same-origin')
        // Without this the isolated page refuses its own worker and wasm.
        headers.set('Cross-Origin-Resource-Policy', 'same-origin')

        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers,
        })
      }),
    )
  })
} else {
  const scriptUrl = document.currentScript.src
  const RELOAD_FLAG = 'coi-reloaded'

  ;(async () => {
    if (window.crossOriginIsolated || !window.isSecureContext) {
      return
    }

    if (!navigator.serviceWorker) {
      return
    }

    try {
      const registration = await navigator.serviceWorker.register(scriptUrl)

      // Reload once so the new worker controls this page. The flag stops a
      // boot loop if isolation still fails for some other reason.
      if (!navigator.serviceWorker.controller) {
        await registration.update()

        if (!sessionStorage.getItem(RELOAD_FLAG)) {
          sessionStorage.setItem(RELOAD_FLAG, '1')
          window.location.reload()
        }

        return
      }

      sessionStorage.removeItem(RELOAD_FLAG)
    } catch {
      // Isolation is an optimisation; the simulation falls back to one thread.
    }
  })()
}
