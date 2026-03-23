#!/usr/bin/env node

/**
 * Generates public/sitemap.xml for smartfit.academy.
 * Run automatically as part of the build process (see package.json).
 *
 * Static public routes are listed here. Dynamic routes (e.g. /lessons/:id)
 * are user-generated and excluded because their URLs are not known at build time.
 * Protected routes (/dashboard, /onboarding, /style-guide) are also excluded.
 */

import { writeFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const BASE_URL = 'https://smartfit.academy';
const TODAY = new Date().toISOString().split('T')[0];

const routes = [
  { path: '/',                      changefreq: 'weekly',  priority: '1.0' },
  { path: '/pricing',               changefreq: 'monthly', priority: '0.8' },
  { path: '/modern-relevance',      changefreq: 'monthly', priority: '0.6' },
  { path: '/netflix-and-nietzsche', changefreq: 'monthly', priority: '0.6' },
];

function buildSitemap(routes) {
  const urls = routes.map(({ path, changefreq, priority }) => `
  <url>
    <loc>${BASE_URL}${path}</loc>
    <lastmod>${TODAY}</lastmod>
    <changefreq>${changefreq}</changefreq>
    <priority>${priority}</priority>
  </url>`).join('');

  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${urls}
</urlset>`;
}

const sitemap = buildSitemap(routes);
const outPath = resolve(__dirname, '../public/sitemap.xml');
writeFileSync(outPath, sitemap, 'utf-8');

console.log(`Sitemap written to ${outPath} (${routes.length} URLs)`);
