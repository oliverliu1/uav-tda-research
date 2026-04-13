/**
 * export_barcode.mjs
 *
 * Exports two research poster graphics as high-resolution PNGs:
 *   1. betti_barcode.png      — Topological Fingerprints (barcode + attack diagram per type)
 *   2. vietoris_rips.png      — Vietoris-Rips filtration construction (3-panel)
 *
 * Usage:
 *   npx puppeteer browsers install chrome   # first time only
 *   node export_barcode.mjs
 */

import puppeteer from "puppeteer";

// ============================================================================
// SHARED HELPERS
// ============================================================================

async function renderToFile(page, html, selector, outPath) {
  await page.setViewport({ width: 1600, height: 900, deviceScaleFactor: 2 });
  await page.setContent(html, { waitUntil: "domcontentloaded" });

  const dims = await page.evaluate((sel) => {
    const el = document.querySelector(sel);
    const r  = el.getBoundingClientRect();
    return { width: Math.ceil(r.width + 40), height: Math.ceil(r.height + 40) };
  }, selector);

  await page.setViewport({ width: dims.width, height: dims.height, deviceScaleFactor: 2 });
  await page.setContent(html, { waitUntil: "domcontentloaded" });

  const el = await page.$(selector);
  await el.screenshot({ path: outPath, omitBackground: false });
  console.log(`✓ Saved: ${outPath}  (${dims.width * 2}×${dims.height * 2}px @2x)`);
}

// ============================================================================
// GRAPHIC 1 — TOPOLOGICAL FINGERPRINTS (BettiBarcode + AttackDiagram)
// ============================================================================

function generateBarcode(type) {
  const seed = { normal: 0.12, sybil: 0.08, blackhole: 0.05, wormhole: 0.18, flooding: 0.03 };
  const r = seed[type];
  return {
    normal:    { beta0: Array.from({ length: 8  }, (_, i) => ({ birth: 0.1  + i*0.08, death: 0.3  + i*0.08 + r*0.15 })), beta1: Array.from({ length: 5  }, (_, i) => ({ birth: 0.2  + i*0.12, death: 0.4  + i*0.12 + r*0.2  })) },
    sybil:     { beta0: Array.from({ length: 18 }, (_, i) => ({ birth: 0.05 + i*0.04, death: 0.5  + i*0.04 + r*0.3  })), beta1: Array.from({ length: 4  }, (_, i) => ({ birth: 0.25 + i*0.15, death: 0.35 + i*0.15 + r*0.1  })) },
    blackhole: { beta0: Array.from({ length: 15 }, (_, i) => ({ birth: 0.1  + i*0.05, death: 0.2  + i*0.05 + r*0.08 })), beta1: Array.from({ length: 2  }, (_, i) => ({ birth: 0.3  + i*0.2,  death: 0.35 + i*0.2  + r*0.05 })) },
    wormhole:  { beta0: Array.from({ length: 7  }, (_, i) => ({ birth: 0.1  + i*0.1,  death: 0.25 + i*0.1  + r*0.12 })), beta1: Array.from({ length: 9  }, (_, i) => ({ birth: 0.05 + i*0.08, death: 0.4  + i*0.08 + r*0.25 })) },
    flooding:  { beta0: Array.from({ length: 25 }, (_, i) => ({ birth: 0.02 + i*0.02, death: 0.06 + i*0.02 + r*0.03 })), beta1: Array.from({ length: 12 }, (_, i) => ({ birth: 0.1  + i*0.04, death: 0.13 + i*0.04 + r*0.04 })) },
  }[type];
}

function renderBettiBarcode(type) {
  const { beta0, beta1 } = generateBarcode(type);
  const W = 180, H = 120;
  const maxP = Math.max(...beta0.map(b => b.death), ...beta1.map(b => b.death));
  const b0 = beta0.map((bar, i) => {
    const y = (i / beta0.length) * (H * 0.45), x1 = (bar.birth / maxP) * (W-20) + 10, bw = ((bar.death-bar.birth) / maxP) * (W-20);
    return `<line x1="${x1.toFixed(2)}" y1="${(y+10).toFixed(2)}" x2="${(x1+bw).toFixed(2)}" y2="${(y+10).toFixed(2)}" stroke="#2563eb" stroke-width="2.5" opacity="0.8"/>`;
  }).join("");
  const b1 = beta1.map((bar, i) => {
    const y = (i / beta1.length) * (H * 0.45) + H*0.5, x1 = (bar.birth / maxP) * (W-20) + 10, bw = ((bar.death-bar.birth) / maxP) * (W-20);
    return `<line x1="${x1.toFixed(2)}" y1="${(y+10).toFixed(2)}" x2="${(x1+bw).toFixed(2)}" y2="${(y+10).toFixed(2)}" stroke="#dc2626" stroke-width="2.5" opacity="0.8"/>`;
  }).join("");
  return `<svg width="${W}" height="${H}" style="background:#fafafa;border:1px solid #e0e0e0;display:block;">${b0}${b1}
    <text x="5" y="25" font-size="10" fill="#2563eb" font-weight="600">β₀</text>
    <text x="5" y="${H*0.5+25}" font-size="10" fill="#dc2626" font-weight="600">β₁</text></svg>`;
}

const ATTACK_DIAGRAMS = {
  normal: `<svg width="180" height="120" viewBox="0 0 180 120">
    <circle cx="90" cy="52" r="45" fill="#10b981" opacity="0.1" stroke="#10b981" stroke-width="2"/>
    ${Array.from({length:8},(_,i)=>{const a=(i/8)*Math.PI*2,x=(90+Math.cos(a)*25).toFixed(1),y=(52+Math.sin(a)*25).toFixed(1);return`<circle cx="${x}" cy="${y}" r="4" fill="#10b981"/><line x1="${x}" y1="${y}" x2="90" y2="52" stroke="#10b981" stroke-width="1" opacity="0.3"/>`;}).join("")}
    <text x="90" y="110" font-size="9" text-anchor="middle" fill="#059669" font-weight="600">Connected swarm</text></svg>`,

  sybil: `<svg width="180" height="120" viewBox="0 0 180 120">
    ${Array.from({length:6},(_,i)=>{const a=(i/6)*Math.PI*2,x=(90+Math.cos(a)*20).toFixed(1),y=(50+Math.sin(a)*20).toFixed(1);return`<circle cx="${x}" cy="${y}" r="4" fill="#10b981"/>`;}).join("")}
    ${Array.from({length:10},(_,i)=>{const a=(i/10)*Math.PI*2,x=(90+Math.cos(a)*35).toFixed(1),y=(50+Math.sin(a)*35).toFixed(1);return`<circle cx="${x}" cy="${y}" r="3" fill="#ef4444" opacity="0.7"/>`;}).join("")}
    <text x="90" y="110" font-size="9" text-anchor="middle" fill="#dc2626" font-weight="600">Forged identities (β₀↑)</text></svg>`,

  blackhole: `<svg width="180" height="120" viewBox="0 0 180 120">
    <circle cx="50" cy="40" r="3" fill="#10b981"/><circle cx="55" cy="45" r="3" fill="#10b981"/><circle cx="60" cy="38" r="3" fill="#10b981"/>
    <circle cx="120" cy="45" r="3" fill="#10b981"/><circle cx="115" cy="50" r="3" fill="#10b981"/>
    <circle cx="90" cy="70" r="3" fill="#10b981"/><circle cx="85" cy="75" r="3" fill="#10b981"/><circle cx="95" cy="73" r="3" fill="#10b981"/>
    <circle cx="90" cy="35" r="6" fill="#ef4444"/>
    <text x="90" y="30" font-size="12" text-anchor="middle" fill="#ef4444" font-weight="700">×</text>
    <text x="90" y="110" font-size="9" text-anchor="middle" fill="#dc2626" font-weight="600">Packet drops (β₀↑, β₁↓)</text></svg>`,

  wormhole: `<svg width="180" height="120" viewBox="0 0 180 120">
    <circle cx="40" cy="50" r="4" fill="#10b981"/><circle cx="50" cy="50" r="4" fill="#10b981"/><circle cx="45" cy="58" r="4" fill="#10b981"/>
    <circle cx="130" cy="50" r="4" fill="#10b981"/><circle cx="140" cy="50" r="4" fill="#10b981"/><circle cx="135" cy="58" r="4" fill="#10b981"/>
    <path d="M 50 50 Q 90 20, 130 50" stroke="#ef4444" stroke-width="2" fill="none" stroke-dasharray="3,3" opacity="0.8"/>
    <text x="90" y="110" font-size="9" text-anchor="middle" fill="#dc2626" font-weight="600">Network β₁ ≠ Physical proximity</text></svg>`,

  flooding: `<svg width="180" height="120" viewBox="0 0 180 120">
    ${[[72,38],[104,29],[118,52],[96,68],[78,61],[60,47],[84,33],[110,41],[66,55],[100,44],[88,25],[114,63],[74,72],[92,56],[58,38],[106,57],[82,44],[70,64],[98,35],[112,48],[76,52],[86,67],[102,31],[64,43],[90,50]].map(([x,y])=>`<circle cx="${x}" cy="${y}" r="2" fill="#ef4444" opacity="0.6"/>`).join("")}
    <circle cx="90" cy="50" r="5" fill="#dc2626"/>
    <text x="90" y="110" font-size="9" text-anchor="middle" fill="#dc2626" font-weight="600">Traffic surge (short bars)</text></svg>`,
};

function buildFingerprintsHTML() {
  const types = ["normal","sybil","blackhole","wormhole","flooding"];
  const titles = { normal:"Normal Traffic", sybil:"Sybil Attack", blackhole:"Blackhole Attack", wormhole:"Wormhole Attack", flooding:"Flooding Attack" };
  const colors = { normal:"#059669", sybil:"#dc2626", blackhole:"#dc2626", wormhole:"#dc2626", flooding:"#dc2626" };
  const cards = types.map(t => `
    <div style="background:white;border-radius:6px;padding:12px;border:1px solid #e2e8f0;">
      <div style="font-size:13px;font-weight:700;margin-bottom:10px;color:${colors[t]};text-align:center;">${titles[t]}</div>
      ${renderBettiBarcode(t)}
      <div style="margin-top:12px;">${ATTACK_DIAGRAMS[t]}</div>
    </div>`).join("");
  return `<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    *{box-sizing:border-box;margin:0;padding:0;}
    body{font-family:-apple-system,sans-serif;background:#f8fafc;padding:20px;display:inline-block;}
    #container{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;background:#f8fafc;padding:16px;border-radius:8px;border:2px solid #e2e8f0;}
  </style></head><body><div id="container">${cards}</div></body></html>`;
}

// ============================================================================
// GRAPHIC 2 — VIETORIS-RIPS CONSTRUCTION (3-panel)
// ============================================================================

function buildVietorisRipsHTML() {
  const PW = 270, PH = 200;

  // ── Deliberately designed point layout ─────────────────────────────────
  //
  // β₀ cluster  (indices 0-2): tight triangle, top-left
  //   All three pairs ~52px apart → all edges form, filled triangle → one component
  //
  // β₁ loop     (indices 3-6): rectangle, centre-right
  //   Side length ~58px (edges form), diagonal ~82px (NO edge) → hollow square = 1-cycle
  //
  // Isolated pt (index 7): top-right corner, far from everything → second β₀ component
  //
  // BALL_R = 32  →  2*BALL_R = 64px threshold
  //   cluster sides ≈ 52  < 64 ✓   cluster diagonals ≈ 74 > 64 ✗  (but 3rd side closes it)
  //   loop sides    ≈ 58  < 64 ✓   loop diagonals    ≈ 82 > 64 ✗  → hole preserved

  const pts = [
    // β₀ cluster
    { x: 52,  y: 60  },  // 0
    { x: 100, y: 42  },  // 1
    { x: 100, y: 88  },  // 2  (triangle: sides ≈ 52-55px, diagonal 46px — all connect)
    // β₁ loop (rectangle, sides 58px, diagonals 82px)
    { x: 162, y: 44  },  // 3  top-left of rect
    { x: 220, y: 44  },  // 4  top-right
    { x: 220, y: 102 },  // 5  bottom-right
    { x: 162, y: 102 },  // 6  bottom-left
    // Isolated point
    { x: 245, y: 22  },  // 7
  ];

  const BALL_R = 32;
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  // Compute edges and triangles from geometry
  const edges = [];
  for (let i = 0; i < pts.length; i++)
    for (let j = i + 1; j < pts.length; j++)
      if (dist(pts[i], pts[j]) < 2 * BALL_R) edges.push([i, j]);

  const tris = [];
  for (let i = 0; i < pts.length; i++)
    for (let j = i + 1; j < pts.length; j++)
      for (let k = j + 1; k < pts.length; k++)
        if (dist(pts[i],pts[j]) < 2*BALL_R && dist(pts[j],pts[k]) < 2*BALL_R && dist(pts[i],pts[k]) < 2*BALL_R)
          tris.push([i, j, k]);

  const f = n => n.toFixed(2);
  const dotsSVG = pts.map(p => `<circle cx="${f(p.x)}" cy="${f(p.y)}" r="4.5" fill="#374151"/>`).join("");

  // ── Panel 1: bare point cloud ─────────────────────────────────────────
  const panel1 = `<svg width="${PW}" height="${PH}" style="background:#fafafa;border:1px solid #e0e0e0;display:block;">${dotsSVG}</svg>`;

  // ── Panel 2: filtration + annotations ────────────────────────────────
  const triSVG  = tris.map(([i,j,k]) =>
    `<polygon points="${f(pts[i].x)},${f(pts[i].y)} ${f(pts[j].x)},${f(pts[j].y)} ${f(pts[k].x)},${f(pts[k].y)}" fill="#2563eb" fill-opacity="0.12" stroke="none"/>`
  ).join("");
  const ballSVG = pts.map(p =>
    `<circle cx="${f(p.x)}" cy="${f(p.y)}" r="${BALL_R}" fill="none" stroke="#2563eb" stroke-width="1.5" stroke-opacity="0.28"/>`
  ).join("");
  const edgeSVG = edges.map(([i,j]) =>
    `<line x1="${f(pts[i].x)}" y1="${f(pts[i].y)}" x2="${f(pts[j].x)}" y2="${f(pts[j].y)}" stroke="#2563eb" stroke-width="2" stroke-opacity="0.70"/>`
  ).join("");

  // β₀ annotation: soft blue ellipse encircling the cluster (pts 0-2)
  // Cluster centre ≈ (84, 63), needs to comfortably wrap the three nodes
  const b0Annot = `
    <ellipse cx="84" cy="65" rx="52" ry="40"
      fill="#2563eb" fill-opacity="0.07"
      stroke="#2563eb" stroke-width="1.5" stroke-dasharray="4,3" stroke-opacity="0.55"/>
    <text x="84" y="20" font-size="9" fill="#2563eb" font-weight="700" text-anchor="middle">β₀ component</text>
    <line x1="84" y1="22" x2="84" y2="28" stroke="#2563eb" stroke-width="1" stroke-opacity="0.6"/>`;

  // β₁ annotation: dotted red loop tracing the rectangular hole
  // Rectangle corners: (162,44) (220,44) (220,102) (162,102)
  // Loop path traces slightly inside the square to make it look like the hole itself
  const loopCx = 191, loopCy = 73, rx = 36, ry = 34;
  const b1Annot = `
    <ellipse cx="${loopCx}" cy="${loopCy}" rx="${rx}" ry="${ry}"
      fill="none"
      stroke="#dc2626" stroke-width="2" stroke-dasharray="5,3" stroke-opacity="0.75"/>
    <text x="${loopCx}" y="${loopCy + ry + 18}" font-size="9" fill="#dc2626" font-weight="700" text-anchor="middle">β₁ loop</text>`;

  const panel2 = `<svg width="${PW}" height="${PH}" style="background:#fafafa;border:1px solid #e0e0e0;display:block;">
    ${triSVG}${ballSVG}${edgeSVG}${dotsSVG}
    ${b0Annot}
    ${b1Annot}
  </svg>`;

  // Barcode panel
  const PAD = {l:28, r:12, t:10, b:30};
  const barW = PW - PAD.l - PAD.r, barH = PH - PAD.t - PAD.b;
  const b0H = barH * 0.52, b1H = barH * 0.40, b1Y = PAD.t + b0H + barH*0.08;
  const toX = v => PAD.l + v * barW;
  const B0 = [{birth:0,death:1},{birth:0,death:0.22},{birth:0,death:0.31},{birth:0,death:0.38},{birth:0,death:0.15},{birth:0,death:0.09},{birth:0,death:0.45}];
  const B1 = [{birth:0.30,death:0.88},{birth:0.42,death:0.60},{birth:0.55,death:0.65}];

  const b0bars = B0.map((bar,i)=>{
    const y = PAD.t + (i/B0.length)*b0H + b0H/B0.length/2;
    return `<line x1="${toX(bar.birth).toFixed(2)}" y1="${y.toFixed(2)}" x2="${toX(bar.death).toFixed(2)}" y2="${y.toFixed(2)}" stroke="#2563eb" stroke-width="2.5" stroke-opacity="0.85"/>`;
  }).join("");
  const b1bars = B1.map((bar,i)=>{
    const y = b1Y + (i/B1.length)*b1H + b1H/B1.length/2;
    return `<line x1="${toX(bar.birth).toFixed(2)}" y1="${y.toFixed(2)}" x2="${toX(bar.death).toFixed(2)}" y2="${y.toFixed(2)}" stroke="#dc2626" stroke-width="2.5" stroke-opacity="0.85"/>`;
  }).join("");

  const panel3 = `<svg width="${PW}" height="${PH}" style="background:#fafafa;border:1px solid #e0e0e0;display:block;">
    <text x="14" y="${PAD.t + b0H/2 + 4}" font-size="10" fill="#2563eb" font-weight="700" text-anchor="middle" transform="rotate(-90,14,${PAD.t + b0H/2})">β₀</text>
    ${b0bars}
    <line x1="${PAD.l}" y1="${b1Y-4}" x2="${PW-PAD.r}" y2="${b1Y-4}" stroke="#e0e0e0" stroke-width="1"/>
    <text x="14" y="${b1Y + b1H/2 + 4}" font-size="10" fill="#dc2626" font-weight="700" text-anchor="middle" transform="rotate(-90,14,${b1Y + b1H/2})">β₁</text>
    ${b1bars}
    <line x1="${PAD.l}" y1="${PH-PAD.b+4}" x2="${PW-PAD.r}" y2="${PH-PAD.b+4}" stroke="#374151" stroke-width="1.2"/>
    <text x="${PAD.l + barW/2}" y="${PH-PAD.b+16}" font-size="9" fill="#6b7280" text-anchor="middle">Filtration parameter ε →</text>
  </svg>`;

  const panelStyle = "display:flex;flex-direction:column;align-items:center;gap:6px;";
  const titleStyle = "font-size:11px;font-weight:700;color:#1a1a1a;letter-spacing:0.03em;text-align:center;";
  const labelStyle = `font-size:10px;color:#6b7280;text-align:center;max-width:${PW}px;line-height:1.4;`;
  const arrowStyle = "display:flex;align-items:center;align-self:center;color:#9ca3af;font-size:20px;padding-bottom:28px;";

  return `<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    *{box-sizing:border-box;margin:0;padding:0;}
    body{font-family:-apple-system,sans-serif;background:#ffffff;padding:20px;display:inline-block;}
    #container{display:flex;align-items:flex-start;gap:8px;}
  </style></head><body><div id="container">
    <div style="${panelStyle}">
      <div style="${titleStyle}">Point Cloud (ε = 0)</div>
      ${panel1}
      <div style="${labelStyle}">Network flows as points in high-dimensional space</div>
    </div>
    <div style="${arrowStyle}">→</div>
    <div style="${panelStyle}">
      <div style="${titleStyle}">Growing Filtration (ε = r)</div>
      ${panel2}
      <div style="${labelStyle}">Vietoris-Rips: connect points within distance ε</div>
    </div>
    <div style="${arrowStyle}">→</div>
    <div style="${panelStyle}">
      <div style="${titleStyle}">Persistence Barcode</div>
      ${panel3}
      <div style="${labelStyle}">Barcode captures topological features across all scales</div>
    </div>
  </div></body></html>`;
}

// ============================================================================
// MAIN
// ============================================================================

const browser = await puppeteer.launch({ headless: "new" });
const page    = await browser.newPage();

await renderToFile(page, buildFingerprintsHTML(),  "#container", "betti_barcode.png");
await renderToFile(page, buildVietorisRipsHTML(),   "#container", "vietoris_rips.png");

await browser.close();
