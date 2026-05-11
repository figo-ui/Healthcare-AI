/**
 * HealthAI Service Worker — PWA offline support
 * Strategy: cache-first for static assets, network-first for API calls
 */

const CACHE_NAME = 'healthai-v1';
const STATIC_ASSETS = [
    '/',
    '/index.html',
    '/auth.html',
    '/src/css/styles.css',
    '/src/css/auth.css',
    '/src/js/app.js',
    '/src/js/api.js',
    '/src/js/auth.js',
    '/src/js/utils.js',
    '/src/js/features/chat.js',
    '/src/js/features/dashboard.js',
    '/src/js/features/symptoms.js',
    '/src/js/features/facilities.js',
    '/src/js/features/emergency.js',
    '/src/views/chat.html',
    '/src/views/dashboard.html',
    '/src/views/symptoms.html',
    '/src/views/facilities.html',
    '/favicon.svg',
    '/manifest.json',
];

// Install: pre-cache all static assets
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(STATIC_ASSETS).catch((err) => {
                console.warn('[SW] Pre-cache partial failure:', err);
            });
        })
    );
    self.skipWaiting();
});

// Activate: clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) =>
            Promise.all(
                keys
                    .filter((key) => key !== CACHE_NAME)
                    .map((key) => caches.delete(key))
            )
        )
    );
    self.clients.claim();
});

// Fetch: network-first for API, cache-first for static
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // Always go to network for API calls — never serve stale medical data
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(
            fetch(event.request).catch(() => {
                return new Response(
                    JSON.stringify({ error: 'You are offline. Please check your connection.' }),
                    { status: 503, headers: { 'Content-Type': 'application/json' } }
                );
            })
        );
        return;
    }

    // Cache-first for static assets
    event.respondWith(
        caches.match(event.request).then((cached) => {
            if (cached) return cached;
            return fetch(event.request).then((response) => {
                // Cache successful GET responses for static assets
                if (
                    response.ok &&
                    event.request.method === 'GET' &&
                    !url.pathname.startsWith('/admin/')
                ) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
                }
                return response;
            }).catch(() => {
                // Offline fallback for HTML navigation
                if (event.request.headers.get('accept')?.includes('text/html')) {
                    return caches.match('/index.html');
                }
            });
        })
    );
});
