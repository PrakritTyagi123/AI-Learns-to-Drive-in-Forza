/*
 * ForzaTek AI — Shared JavaScript (v3 redesign)
 * ============================================
 * Same public API as v2 — every page calls ForzaTek.mount(pageId) and uses
 * ForzaTek.eel / .api / .fmt / .toast. The only changes from v2 are:
 *
 *   • Auto-injects Google Fonts (Inter + JetBrains Mono + IBM Plex Sans
 *     fallback) so pages don't each need their own <link> tag.
 *   • Slightly refreshed sidebar icons (consistent 1.5 stroke, 16-px viewBox
 *     equivalents).
 *   • Static-preview fallback: when no Eel bridge and no FastAPI side server
 *     respond, ForzaTek.eel() / ForzaTek.api.get() return canned data so the
 *     redesigned UI looks alive in a browser preview. This is silent and
 *     only kicks in when both backends are confirmed unreachable.
 */
(function () {
  'use strict';

  const FASTAPI_BASE = `http://127.0.0.1:8001`;

  // ─── Theme: apply BEFORE first paint ──────────────────────────────────
  // Default = light. Persist user's choice in localStorage. No OS sniff —
  // user explicitly asked for light-by-default.
  (function applyThemeEarly() {
    try {
      const saved = localStorage.getItem('ftk.theme');
      if (saved === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
      }
    } catch (e) { /* localStorage blocked — fine, stay light */ }
  })();

  function setTheme(mode) {
    if (mode === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    try { localStorage.setItem('ftk.theme', mode); } catch (e) {}
  }
  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  }

  // ─── Auto-load fonts ──────────────────────────────────────────────────
  (function loadFonts() {
    if (document.querySelector('link[data-ftk-fonts]')) return;
    const pre1 = document.createElement('link');
    pre1.rel = 'preconnect'; pre1.href = 'https://fonts.googleapis.com';
    const pre2 = document.createElement('link');
    pre2.rel = 'preconnect'; pre2.href = 'https://fonts.gstatic.com';
    pre2.crossOrigin = 'anonymous';
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.setAttribute('data-ftk-fonts', '1');
    link.href =
      'https://fonts.googleapis.com/css2' +
      '?family=Inter:wght@400;500;600' +
      '&family=IBM+Plex+Sans:wght@400;500;600' +
      '&family=JetBrains+Mono:wght@400;500;600' +
      '&display=swap';
    document.head.appendChild(pre1);
    document.head.appendChild(pre2);
    document.head.appendChild(link);
  })();

  // ─── Nav definition ───
  const NAV = [
    { section: 'Overview' },
    { id: 'dashboard',  label: 'Dashboard',  href: 'dashboard.html', icon: 'grid' },

    { section: 'Data' },
    { id: 'ingest',     label: 'Ingest',     href: 'ingest.html',    icon: 'download' },
    { id: 'hud_mask',   label: 'HUD mask',   href: 'hud_mask.html',  icon: 'frame' },
    { id: 'label',      label: 'Label',      href: 'label.html',     icon: 'pen' },

    { section: 'Model' },
    { id: 'train',      label: 'Train',      href: 'train.html',     icon: 'cpu' },
    { id: 'compare',    label: 'Compare',    href: 'compare.html',   icon: 'target' },

    { section: 'Runtime' },
    { id: 'telemetry',  label: 'Telemetry',  href: 'telemetry.html', icon: 'gauge' },
    { id: 'drive',      label: 'Drive',      href: 'drive.html',     icon: 'steering' },
    { id: 'ppo',        label: 'PPO train',  href: 'ppo.html',       icon: 'brain' },

    { section: 'System' },
    { id: 'settings',   label: 'Settings',   href: 'settings.html',  icon: 'sliders' },
    { id: 'help',       label: 'Help',       href: 'help.html',      icon: 'book' },
  ];

  // 16x16 stroke icons, consistent 1.5px stroke.
  const ICON = {
    grid:     '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
    download: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v13M7 11l5 5 5-5M4 21h16"/></svg>',
    frame:    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="1"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18" opacity="0.4"/></svg>',
    pen:      '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 4l6 6L9 21H3v-6L14 4z"/><path d="M13 5l6 6"/></svg>',
    cpu:      '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="5" width="14" height="14" rx="1"/><rect x="9" y="9" width="6" height="6" rx="0.5"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/></svg>',
    target:   '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/></svg>',
    gauge:    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 17a8 8 0 1116 0"/><path d="M12 17l4-6"/><circle cx="12" cy="17" r="1.2" fill="currentColor"/></svg>',
    steering: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="2"/><path d="M12 3v7M4.5 8.5l6 4M19.5 8.5l-6 4M7.5 20l3-6M16.5 20l-3-6"/></svg>',
    brain:    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 4a3 3 0 00-3 3 3 3 0 00-2 5 3 3 0 002 5 3 3 0 003 3h1V4H9z"/><path d="M15 4a3 3 0 013 3 3 3 0 012 5 3 3 0 01-2 5 3 3 0 01-3 3h-1V4h1z"/></svg>',
    sliders:  '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/></svg>',
    book:     '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5a2 2 0 012-2h12v16H6a2 2 0 00-2 2V5z"/><path d="M4 19h14M8 7h7M8 11h5"/></svg>',
    chevL:    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 6l-6 6 6 6"/></svg>',
    chevR:    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>',
    moon:     '<svg class="icon-moon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z"/></svg>',
    sun:      '<svg class="icon-sun" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>',
  };

  // ─── Sidebar ───
  function renderSidebar(activeId) {
    const collapsed = localStorage.getItem('ftk.sidebar.collapsed') === '1';
    document.body.classList.add('with-sidebar');
    if (collapsed) document.body.classList.add('sidebar-collapsed');

    const items = NAV.map(item => {
      if (item.section) {
        return `<div class="sidebar-section">${item.section}</div>`;
      }
      const cls = item.id === activeId ? 'sidebar-link active' : 'sidebar-link';
      const ic  = ICON[item.icon] || '';
      return `
        <a href="${item.href}" class="${cls}" title="${item.label}">
          ${ic}<span>${item.label}</span>
        </a>`;
    }).join('');

    const sidebar = document.createElement('aside');
    sidebar.className = 'sidebar';
    sidebar.innerHTML = `
      <div class="sidebar-header">
        <div class="sidebar-brand">FORZATEK</div>
        <div class="sidebar-header-actions">
          <button class="theme-toggle" id="ftkThemeToggle"
                  title="Toggle theme" aria-label="Toggle theme">
            ${ICON.moon}${ICON.sun}
          </button>
          <button class="sidebar-toggle" id="ftkSidebarToggle"
                  title="${collapsed ? 'Expand' : 'Collapse'}">
            ${collapsed ? ICON.chevR : ICON.chevL}
          </button>
        </div>
      </div>
      <nav class="sidebar-nav">${items}</nav>
      <div class="sidebar-footer">
        <span class="health-dot" id="ftkHealthDot"></span>
        <span class="health-text" id="ftkHealthText">checking…</span>
      </div>`;
    document.body.appendChild(sidebar);

    document.getElementById('ftkSidebarToggle').addEventListener('click', () => {
      const isCollapsed = document.body.classList.toggle('sidebar-collapsed');
      localStorage.setItem('ftk.sidebar.collapsed', isCollapsed ? '1' : '0');
      const btn = document.getElementById('ftkSidebarToggle');
      btn.innerHTML = isCollapsed ? ICON.chevR : ICON.chevL;
      btn.title = isCollapsed ? 'Expand' : 'Collapse';
    });

    document.getElementById('ftkThemeToggle').addEventListener('click', () => {
      const next = currentTheme() === 'dark' ? 'light' : 'dark';
      setTheme(next);
      // Tell any page-local listeners (charts, canvases) the theme changed.
      window.dispatchEvent(new CustomEvent('ftk:themechange', { detail: { theme: next } }));
    });
  }

  // ─── Health poll ───
  let _healthState = 'unknown';
  function updateHealth(state, text) {
    _healthState = state;
    const dot  = document.getElementById('ftkHealthDot');
    const txt  = document.getElementById('ftkHealthText');
    if (!dot || !txt) return;
    dot.className = 'health-dot ' + (state === 'ok' ? 'ok' : state === 'warn' ? 'warn' : 'bad');
    txt.textContent = text;
  }

  async function pollHealth() {
    try {
      if (typeof eel !== 'undefined' && eel.system_health) {
        const r = await eel.system_health()();
        if (r && r.ok) { updateHealth('ok', 'connected'); return; }
      }
      const r = await fetch(`${FASTAPI_BASE}/api/system/health`);
      if (r.ok) { updateHealth('ok', 'connected'); return; }
      updateHealth('warn', 'degraded');
    } catch (e) {
      // No backend at all — switch to demo mode silently.
      if (DEMO_MODE) updateHealth('warn', 'demo · offline');
      else            updateHealth('bad',  'disconnected');
    }
  }

  // ─── Demo-mode mock data (used only when backends are unreachable) ───
  let DEMO_MODE = false;
  const DEMO = {
    system_stats: {
      total_frames:   18420,
      labeled_frames: 11203,
      proposed_frames: 1872,
      queue_size:     347,
      active_model: { id: 7, name: 'perception_v1', round_num: 12 },
      runtime: {
        capture: { state: 'idle' },
        gamepad: { state: 'disconnected' },
      },
      frames_by_version: { fh4: 4120, fh5: 11890, fh6: 2410 },
    },
  };

  // ─── Eel/REST helpers ───
  function callEel(name, ...args) {
    return new Promise((resolve, reject) => {
      if (typeof eel === 'undefined' || typeof eel[name] !== 'function') {
        if (DEMO_MODE && DEMO[name]) return resolve({ ok: true, data: DEMO[name] });
        return reject(new Error(`Eel function '${name}' is not available`));
      }
      try {
        eel[name](...args)((result) => resolve(result));
      } catch (err) {
        reject(err);
      }
    });
  }

  async function apiGet(path) {
    try {
      const r = await fetch(`${FASTAPI_BASE}${path}`);
      if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
      return r.json();
    } catch (e) {
      // Last-resort demo fallback
      if (path.includes('/api/system/stats') && DEMO[`system_stats`]) {
        DEMO_MODE = true;
        return DEMO.system_stats;
      }
      throw e;
    }
  }

  async function apiPost(path, body) {
    const r = await fetch(`${FASTAPI_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    if (!r.ok) throw new Error(`POST ${path} → ${r.status}`);
    return r.json();
  }

  function fastapiBase() { return FASTAPI_BASE; }

  // ─── Formatters ───
  const fmt = {
    int(n) {
      if (n === null || n === undefined || Number.isNaN(n)) return '—';
      return Number(n).toLocaleString('en-US');
    },
    bytes(n) {
      if (!n) return '0 B';
      const u = ['B', 'KB', 'MB', 'GB', 'TB'];
      let i = 0;
      while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
      return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
    },
    timeAgo(ts) {
      if (!ts) return '—';
      const s = Math.max(0, Math.floor(Date.now() / 1000 - ts));
      if (s < 60)    return `${s}s ago`;
      if (s < 3600)  return `${Math.floor(s / 60)}m ago`;
      if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
      return `${Math.floor(s / 86400)}d ago`;
    },
  };

  // ─── Toasts ───
  function toast(message, kind) {
    kind = kind || 'ok';
    if (kind === 'danger') kind = 'bad';
    let host = document.getElementById('ftkToastHost');
    if (!host) {
      host = document.createElement('div');
      host.id = 'ftkToastHost';
      Object.assign(host.style, {
        position: 'fixed', bottom: '20px', right: '20px',
        display: 'flex', flexDirection: 'column', gap: '8px',
        zIndex: 10000, pointerEvents: 'none',
      });
      document.body.appendChild(host);
    }
    const el = document.createElement('div');
    const palette = {
      ok:   ['var(--ok)', 'var(--ok-bg)'],
      warn: ['var(--warn)', 'var(--warn-bg)'],
      bad:  ['var(--danger)', 'var(--danger-bg)'],
    }[kind] || ['var(--ink-dim)', 'var(--surface-2)'];
    Object.assign(el.style, {
      padding: '10px 14px',
      border: `1px solid ${palette[0]}`,
      background: palette[1],
      color: palette[0],
      borderRadius: '6px',
      fontFamily: 'var(--font-mono)',
      fontSize: '11.5px',
      letterSpacing: '0.04em',
      maxWidth: '340px',
      backdropFilter: 'blur(8px)',
    });
    el.textContent = message;
    host.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity 0.4s'; }, 2400);
    setTimeout(() => { el.remove(); }, 2900);
  }

  // ─── Probe demo mode once at boot ───
  async function probeDemoMode() {
    if (typeof eel !== 'undefined' && eel.system_health) return; // real backend
    try {
      const r = await fetch(`${FASTAPI_BASE}/api/system/health`, {
        signal: AbortSignal.timeout(800),
      });
      if (r.ok) return; // real backend
    } catch (e) { /* fall through */ }
    DEMO_MODE = true;
  }

  // ─── Public ───
  function mount(pageId) {
    renderSidebar(pageId);
    probeDemoMode().then(() => {
      pollHealth();
      setInterval(pollHealth, 3000);
    });
  }

  window.ForzaTek = {
    mount,
    eel: callEel,
    api: { get: apiGet, post: apiPost },
    fastapiBase,
    fmt,
    toast,
    setTheme,
    currentTheme,
    get demoMode() { return DEMO_MODE; },
  };
})();
