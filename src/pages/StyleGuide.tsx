import React from 'react';
import '../css/index.css';

const colorTokens = [
  { name: '--color-primary', desc: 'Primary', var: 'var(--color-primary)' },
  { name: '--color-secondary', desc: 'Secondary', var: 'var(--color-secondary)' },
  { name: '--color-accent', desc: 'Accent', var: 'var(--color-accent)' },
  { name: '--color-success', desc: 'Success', var: 'var(--color-success)' },
  { name: '--color-warning', desc: 'Warning', var: 'var(--color-warning)' },
  { name: '--color-danger', desc: 'Danger', var: 'var(--color-danger)' },
  { name: '--color-background', desc: 'Background', var: 'var(--color-background)' },
  { name: '--color-surface', desc: 'Surface', var: 'var(--color-surface)' },
  { name: '--color-card', desc: 'Card', var: 'var(--color-card)' },
  { name: '--color-muted', desc: 'Muted', var: 'var(--color-muted)' },
  { name: '--color-code-bg', desc: 'Code BG', var: 'var(--color-code-bg)' },
];

const fontTokens = [
  { name: '--font-sans', desc: 'Sans', var: 'var(--font-sans)' },
  { name: '--font-mono', desc: 'Mono', var: 'var(--font-mono)' },
];

const radiusTokens = [
  { name: '--radius-xs', desc: 'XS', val: 'var(--radius-xs)' },
  { name: '--radius-sm', desc: 'SM', val: 'var(--radius-sm)' },
  { name: '--radius-md', desc: 'MD', val: 'var(--radius-md)' },
  { name: '--radius-lg', desc: 'LG', val: 'var(--radius-lg)' },
  { name: '--radius-xl', desc: 'XL', val: 'var(--radius-xl)' },
];

const shadowTokens = [
  { name: '--shadow-xs', desc: 'XS', val: 'var(--shadow-xs)' },
  { name: '--shadow-sm', desc: 'SM', val: 'var(--shadow-sm)' },
  { name: '--shadow-md', desc: 'MD', val: 'var(--shadow-md)' },
  { name: '--shadow-lg', desc: 'LG', val: 'var(--shadow-lg)' },
];

const spacingTokens = [
  { name: '--space-xs', desc: 'XS', val: 'var(--space-xs)' },
  { name: '--space-sm', desc: 'SM', val: 'var(--space-sm)' },
  { name: '--space-md', desc: 'MD', val: 'var(--space-md)' },
  { name: '--space-lg', desc: 'LG', val: 'var(--space-lg)' },
  { name: '--space-xl', desc: 'XL', val: 'var(--space-xl)' },
];

const StyleGuide = () => (
  <div style={{ maxWidth: 900, margin: '0 auto', padding: '2rem' }}>
    <h1 className="headline">Style Guide</h1>

    {/* Color Tokens */}
    <section style={{ marginBottom: '2rem' }}>
      <h2 className="text-title mb-md">Color Tokens</h2>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem' }}>
        {colorTokens.map((c) => (
          <div key={c.name} style={{ background: c.var, width: 100, height: 60, borderRadius: 8, boxShadow: 'var(--shadow-xs)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#222', border: '1px solid #e5e7eb' }}>
            <div style={{ fontSize: 12, fontWeight: 600 }}>{c.desc}</div>
            <div style={{ fontSize: 10 }}>{c.name}</div>
          </div>
        ))}
      </div>
    </section>

    {/* Typography */}
    <section className="mb-md">
      <h2 className="text-title mb-md">Typography</h2>
      <div style={{ display: 'flex', gap: '2rem', alignItems: 'flex-end' }}>
        <div>
          <div className="headline" style={{ fontFamily: 'var(--font-sans)' }}>Headline Example</div>
          <div className="text-title" style={{ fontFamily: 'var(--font-sans)' }}>Title Example</div>
          <div className="text-body" style={{ fontFamily: 'var(--font-sans)' }}>Body text example</div>
        </div>
        <div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--font-size-base)' }}>Mono Example: <span style={{ fontWeight: 700 }}>1234567890</span></div>
        </div>
      </div>
    </section>

    {/* Border Radius & Shadow */}
    <section className="mb-md">
      <h2 className="text-title mb-md">Border Radius & Shadows</h2>
      <div style={{ display: 'flex', gap: '2rem' }}>
        <div>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {radiusTokens.map((r) => (
              <div key={r.name} style={{ width: 50, height: 50, background: 'var(--color-card)', border: '1px solid var(--color-border)', borderRadius: r.val, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10 }}>{r.desc}</div>
            ))}
          </div>
          <div style={{ fontSize: 12, marginTop: 4 }}>Radius</div>
        </div>
        <div>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {shadowTokens.map((s) => (
              <div key={s.name} style={{ width: 50, height: 50, background: 'var(--color-card)', border: '1px solid var(--color-border)', borderRadius: 8, boxShadow: s.val, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10 }}>{s.desc}</div>
            ))}
          </div>
          <div style={{ fontSize: 12, marginTop: 4 }}>Shadow</div>
        </div>
      </div>
    </section>

    {/* Spacing */}
    <section className="mb-md">
      <h2 className="text-title mb-md">Spacing Tokens</h2>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        {spacingTokens.map((s) => (
          <div key={s.name} style={{ background: 'var(--color-card)', border: '1px solid var(--color-border)', height: 18, width: `calc(${s.val} * 6)`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10 }}>{s.desc}</div>
        ))}
      </div>
    </section>

    {/* Card Example */}
    <section className="mb-md">
      <h2 className="text-title mb-md">Card Component</h2>
      <div className="card" style={{ maxWidth: 350 }}>
        <div className="card-title">Card Title</div>
        <div className="card-content">This is an example of a card component using the design system tokens and composite classes.</div>
      </div>
    </section>

    {/* Headline Example */}
    <section className="mb-md">
      <h2 className="text-title mb-md">Headline Example</h2>
      <div className="headline">This is a Headline</div>
    </section>

    {/* Code Block Example */}
    <section className="mb-md">
      <h2 className="text-title mb-md">Code Block Example</h2>
      <pre className="code-block">{`
function greet(name) {
  return 'Hello, ' + name + '!';
}
`}</pre>
    </section>

    {/* Utility Classes Example */}
    <section className="mb-md">
      <h2 className="text-title mb-md">Utility Classes</h2>
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
        <div className="bg-primary px-md py-md rounded text-body">.bg-primary</div>
        <div className="bg-secondary px-md py-md rounded text-body">.bg-secondary</div>
        <div className="bg-accent px-md py-md rounded text-body">.bg-accent</div>
        <div className="bg-card px-md py-md rounded text-body">.bg-card</div>
        <div className="text-primary">.text-primary</div>
        <div className="text-secondary">.text-secondary</div>
        <div className="text-accent">.text-accent</div>
        <div className="text-muted">.text-muted</div>
        <div className="shadow px-md py-md rounded">.shadow</div>
        <div className="shadow-lg px-md py-md rounded">.shadow-lg</div>
      </div>
    </section>
  </div>
);

export default StyleGuide;
