import { Hono } from 'hono'
import type { AppContext } from './types'

const analyticsRoutes = new Hono<AppContext>()

function parseUserAgent(ua: string): { device_type: string; browser: string; os: string } {
  // Device type
  let device_type = 'desktop'
  if (/Tablet|iPad/i.test(ua)) {
    device_type = 'tablet'
  } else if (/Mobi|Android|iPhone/i.test(ua)) {
    device_type = 'mobile'
  }

  // Browser — order matters: Edge must be checked before Chrome
  let browser = 'other'
  if (/Edg\//i.test(ua)) {
    browser = 'edge'
  } else if (/Chrome\//i.test(ua)) {
    browser = 'chrome'
  } else if (/Firefox\//i.test(ua)) {
    browser = 'firefox'
  } else if (/Safari\//i.test(ua)) {
    browser = 'safari'
  }

  // OS
  let os = 'other'
  if (/iPhone|iPad|iOS/i.test(ua)) {
    os = 'ios'
  } else if (/Android/i.test(ua)) {
    os = 'android'
  } else if (/Windows/i.test(ua)) {
    os = 'windows'
  } else if (/Mac OS X/i.test(ua)) {
    os = 'macos'
  } else if (/Linux/i.test(ua)) {
    os = 'linux'
  }

  return { device_type, browser, os }
}

// POST /track — public, no auth required
analyticsRoutes.post('/', async (c) => {
  let body: Record<string, unknown>
  try {
    body = await c.req.json()
  } catch {
    return c.json({ error: 'Invalid JSON' }, 400)
  }

  const path = typeof body.path === 'string' ? body.path.slice(0, 500) : null
  if (!path) {
    return c.json({ error: 'Missing path' }, 400)
  }

  // Country from Cloudflare request metadata — no IP stored
  const cf = (c.req.raw as Request & { cf?: Record<string, unknown> }).cf
  const country = typeof cf?.country === 'string' ? cf.country : null

  // User-Agent parsed server-side
  const ua = c.req.header('user-agent') ?? ''
  const { device_type, browser, os } = parseUserAgent(ua)

  const title       = typeof body.title    === 'string' ? body.title.slice(0, 300)    : null
  const referrer    = typeof body.referrer === 'string' ? body.referrer.slice(0, 500) : null
  const language    = typeof body.language === 'string' ? body.language.slice(0, 20)  : null
  const screen_width = typeof body.screen_width === 'number' ? body.screen_width      : null

  const utm_source   = typeof body.utm_source   === 'string' && body.utm_source   ? body.utm_source.slice(0, 100)   : null
  const utm_medium   = typeof body.utm_medium   === 'string' && body.utm_medium   ? body.utm_medium.slice(0, 100)   : null
  const utm_campaign = typeof body.utm_campaign === 'string' && body.utm_campaign ? body.utm_campaign.slice(0, 200) : null
  const utm_term     = typeof body.utm_term     === 'string' && body.utm_term     ? body.utm_term.slice(0, 200)     : null
  const utm_content  = typeof body.utm_content  === 'string' && body.utm_content  ? body.utm_content.slice(0, 200)  : null

  try {
    await c.env.DB.prepare(
      `INSERT INTO page_views
         (path, title, referrer, country, device_type, browser, os, language, screen_width,
          utm_source, utm_medium, utm_campaign, utm_term, utm_content)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).bind(
      path, title, referrer, country, device_type, browser, os, language, screen_width,
      utm_source, utm_medium, utm_campaign, utm_term, utm_content
    ).run()
  } catch (err) {
    console.error('[analytics] D1 insert failed:', err)
    return c.json({ ok: false })
  }

  return c.json({ ok: true })
})

export default analyticsRoutes
