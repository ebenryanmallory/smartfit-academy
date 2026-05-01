import { useEffect } from 'react'
import { useLocation } from 'react-router-dom'

export function usePageTracking() {
  const location = useLocation()

  useEffect(() => {
    // Defer so document.title reflects the newly rendered page
    const timer = setTimeout(() => {
      const url = new URL(window.location.href)
      const payload = {
        path:         location.pathname + location.search,
        title:        document.title,
        referrer:     document.referrer || null,
        language:     navigator.language,
        screen_width: window.screen.width,
        utm_source:   url.searchParams.get('utm_source'),
        utm_medium:   url.searchParams.get('utm_medium'),
        utm_campaign: url.searchParams.get('utm_campaign'),
        utm_term:     url.searchParams.get('utm_term'),
        utm_content:  url.searchParams.get('utm_content'),
      }

      if (localStorage.getItem('analytics_opt_out') === '1') return

      fetch('/track', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        keepalive: true,
      }).catch(() => {})
    }, 0)

    return () => clearTimeout(timer)
  }, [location.pathname, location.search])
}
