<!--
  Better Feed uses its own internal, first-party analytics system rather than
  any third-party tracking service. Page views are captured server-side via a
  POST /track endpoint on the Cloudflare Worker and stored directly in the app's
  Cloudflare D1 database (progressive-ai-academy-db). Tracked data includes page
  path, document title, referrer, country (from Cloudflare request metadata),
  device type, browser, OS, language, screen width, and UTM parameters.
  Use the queries below to retrieve analytics data on demand.
-->

# Analytics Queries

Run any query with:
```bash
npx wrangler d1 execute progressive-ai-academy-db --remote --command "YOUR SQL HERE"
```

---

## Dashboard Overview

**Total pageviews (30 days)**
```bash
npx wrangler d1 execute progressive-ai-academy-db --remote --command "SELECT COUNT(*) AS total_views FROM page_views WHERE viewed_at >= datetime('now', '-30 days');"
```

**Top 10 pages**
```bash
npx wrangler d1 execute progressive-ai-academy-db --remote --command "SELECT path, COUNT(*) AS views FROM page_views WHERE viewed_at >= datetime('now', '-30 days') GROUP BY path ORDER BY views DESC LIMIT 10;"
```

**Daily trend**
```bash
npx wrangler d1 execute progressive-ai-academy-db --remote --command "SELECT date(viewed_at) AS day, COUNT(*) AS views FROM page_views WHERE viewed_at >= datetime('now', '-30 days') GROUP BY day ORDER BY day DESC;"
```

---

## Audience

**Countries**
```bash
npx wrangler d1 execute progressive-ai-academy-db --remote --command "SELECT country, COUNT(*) AS views FROM page_views WHERE country IS NOT NULL AND viewed_at >= datetime('now', '-30 days') GROUP BY country ORDER BY views DESC;"
```

**Languages**
```bash
npx wrangler d1 execute progressive-ai-academy-db --remote --command "SELECT language, COUNT(*) AS views FROM page_views WHERE language IS NOT NULL AND viewed_at >= datetime('now', '-30 days') GROUP BY language ORDER BY views DESC LIMIT 10;"
```

**Device types**
```bash
npx wrangler d1 execute progressive-ai-academy-db --remote --command "SELECT device_type, COUNT(*) AS views FROM page_views WHERE viewed_at >= datetime('now', '-30 days') GROUP BY device_type ORDER BY views DESC;"
```

**Browsers**
```bash
npx wrangler d1 execute progressive-ai-academy-db --remote --command "SELECT browser, COUNT(*) AS views FROM page_views WHERE viewed_at >= datetime('now', '-30 days') GROUP BY browser ORDER BY views DESC;"
```

**Operating systems**
```bash
npx wrangler d1 execute progressive-ai-academy-db --remote --command "SELECT os, COUNT(*) AS views FROM page_views WHERE viewed_at >= datetime('now', '-30 days') GROUP BY os ORDER BY views DESC;"
```

**Screen width breakpoints**
```bash
npx wrangler d1 execute progressive-ai-academy-db --remote --command "SELECT CASE WHEN screen_width < 768 THEN 'mobile (<768)' WHEN screen_width < 1024 THEN 'tablet (768-1023)' ELSE 'desktop (1024+)' END AS breakpoint, COUNT(*) AS views FROM page_views WHERE screen_width IS NOT NULL AND viewed_at >= datetime('now', '-30 days') GROUP BY breakpoint ORDER BY views DESC;"
```

---

## Acquisition

**Referrer sources**
```bash
npx wrangler d1 execute progressive-ai-academy-db --remote --command "SELECT CASE WHEN referrer LIKE '%google%' THEN 'Google' WHEN referrer LIKE '%twitter%' OR referrer LIKE '%t.co%' THEN 'Twitter/X' WHEN referrer LIKE '%linkedin%' THEN 'LinkedIn' WHEN referrer LIKE '%facebook%' THEN 'Facebook' WHEN referrer IS NULL OR referrer = '' THEN 'Direct' ELSE referrer END AS source, COUNT(*) AS views FROM page_views WHERE viewed_at >= datetime('now', '-30 days') GROUP BY source ORDER BY views DESC;"
```

**UTM campaign attribution**
```bash
npx wrangler d1 execute progressive-ai-academy-db --remote --command "SELECT utm_source, utm_medium, utm_campaign, COUNT(*) AS views FROM page_views WHERE utm_source IS NOT NULL AND viewed_at >= datetime('now', '-30 days') GROUP BY utm_source, utm_medium, utm_campaign ORDER BY views DESC;"
```

---

## Content

**Lesson page views**
```bash
npx wrangler d1 execute progressive-ai-academy-db --remote --command "SELECT path, COUNT(*) AS views FROM page_views WHERE path LIKE '/lessons/%' AND viewed_at >= datetime('now', '-30 days') GROUP BY path ORDER BY views DESC;"
```

**Funnel: landing → pricing → onboarding → dashboard**
```bash
npx wrangler d1 execute progressive-ai-academy-db --remote --command "SELECT path, COUNT(*) AS views FROM page_views WHERE path IN ('/', '/pricing', '/onboarding', '/dashboard') AND viewed_at >= datetime('now', '-30 days') GROUP BY path ORDER BY views DESC;"
```
