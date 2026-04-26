/*
 * ForzaTek AI v2 — Shared JavaScript
 * ====================================
 * Loaded by every page.
 *
 * Exposes window.ForzaTek with:
 *   .mount(pageId)      — render sidebar + bind health poll. Call from <body>.
 *   .eel(name, ...args) — invoke an @eel.expose Python function as a Promise.
 *   .api.get(path)      — GET against the FastAPI side server (port 8001).
 *   .api.post(path,bdy) — POST against the FastAPI side server.
 *   .fastapiBase()      — returns the FastAPI side-server URL (for <img> src,
 *                         MJPEG streams, etc. that can't go through Eel).
 *   .fmt.int(n)         — locale-formatted integer.
 *   .fmt.bytes(n)       — pretty bytes.
 *   .fmt.timeAgo(ts)    — "12s ago", "3m ago".
 *   .toast(msg, kind)   — transient banner. kind: ok | warn | bad | danger.
 *
 * The sidebar nav list is the source of truth for menu order. Later modules
 * append to NAV — but for now Module 1 ships with all future pages already
 * listed so the menu is complete from the start; pages that don't exist yet
 * show "coming soon" placeholders.
 */
(function () {
  'use strict';

  const FASTAPI_BASE = `http://127.0.0.1:8001`;

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

  const ICON = {
    grid:     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>',
    download: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 4v12M6 12l6 6 6-6M4 20h16"/></svg>',
    frame:    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="4" y="4" width="16" height="16"/><rect x="9" y="9" width="6" height="6" stroke-dasharray="2 2"/></svg>',
    pen:      '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 20l4-1 10-10-3-3L5 16l-1 4zM13 6l3 3"/></svg>',
    cpu:      '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="6" y="6" width="12" height="12"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/></svg>',
    target:   '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/></svg>',
    gauge:    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 16a8 8 0 1116 0"/><path d="M12 16l4-6"/><circle cx="12" cy="16" r="1" fill="currentColor"/></svg>',
    steering: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="2"/><path d="M12 3v7M3.5 8.5L10 12M20.5 8.5L14 12M7 20l4-6M17 20l-4-6"/></svg>',
    brain:    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M9 4a3 3 0 00-3 3v2a3 3 0 00-1 5.5V17a3 3 0 003 3h1V4H9zM15 4a3 3 0 013 3v2a3 3 0 011 5.5V17a3 3 0 01-3 3h-1V4h0z"/></svg>',
    sliders:  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h14M18 18h2"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="16" cy="18" r="2"/></svg>',
    book:     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 5a2 2 0 012-2h12v16H6a2 2 0 00-2 2V5z"/><path d="M4 19h14"/></svg>',
    chevL:    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 6l-6 6 6 6"/></svg>',
    chevR:    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 6l6 6-6 6"/></svg>',
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
        <div class="sidebar-brand">FORZATEK · v2</div>
        <button class="sidebar-toggle" id="ftkSidebarToggle"
                title="${collapsed ? 'Expand' : 'Collapse'}">
          ${collapsed ? ICON.chevR : ICON.chevL}
        </button>
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
    // Prefer Eel (it's faster + same-origin), fall back to FastAPI side port.
    try {
      if (typeof eel !== 'undefined' && eel.system_health) {
        const r = await eel.system_health()();
        if (r && r.ok) {
          updateHealth('ok', 'connected');
          return;
        }
      }
      const r = await fetch(`${FASTAPI_BASE}/api/system/health`);
      if (r.ok) {
        updateHealth('ok', 'connected');
        return;
      }
      updateHealth('warn', 'degraded');
    } catch (e) {
      updateHealth('bad', 'disconnected');
    }
  }

  // ─── Eel/REST helpers ───
  function callEel(name, ...args) {
    return new Promise((resolve, reject) => {
      if (typeof eel === 'undefined') {
        return reject(new Error('Eel bridge not available — open via the desktop app.'));
      }
      const fn = eel[name];
      if (typeof fn !== 'function') {
        return reject(new Error(`Eel function '${name}' is not exposed`));
      }
      try {
        fn(...args)((result) => resolve(result));
      } catch (err) {
        reject(err);
      }
    });
  }

  async function apiGet(path) {
    const r = await fetch(`${FASTAPI_BASE}${path}`);
    if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
    return r.json();
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

  // FastAPI side-server base URL. Used by pages that need to wire raw
  // <img> / <video> / EventSource directly at FastAPI (e.g. MJPEG preview
  // in Module 2's ingest.html — Eel can't carry binary streams well).
  function fastapiBase() {
    return FASTAPI_BASE;
  }

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
  // Accepts kind: 'ok' | 'warn' | 'bad' | 'danger' (alias of 'bad').
  function toast(message, kind) {
    kind = kind || 'ok';
    if (kind === 'danger') kind = 'bad';   // tolerate the common alias
    let host = document.getElementById('ftkToastHost');
    if (!host) {
      host = document.createElement('div');
      host.id = 'ftkToastHost';
      Object.assign(host.style, {
        position: 'fixed', bottom: '16px', right: '16px',
        display: 'flex', flexDirection: 'column', gap: '6px',
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
      padding: '8px 12px',
      border: `1px solid ${palette[0]}`,
      background: palette[1],
      color: palette[0],
      borderRadius: '3px',
      fontFamily: 'var(--font-mono)',
      fontSize: '11px',
      letterSpacing: '0.04em',
      maxWidth: '320px',
    });
    el.textContent = message;
    host.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity 0.4s'; }, 2400);
    setTimeout(() => { el.remove(); }, 2900);
  }

  // ─── Public ───
  function mount(pageId) {
    renderSidebar(pageId);
    pollHealth();
    setInterval(pollHealth, 3000);
  }

  window.ForzaTek = {
    mount,
    eel: callEel,
    api: { get: apiGet, post: apiPost },
    fastapiBase,
    fmt,
    toast,
  };
})();