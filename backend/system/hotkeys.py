"""
ForzaTek AI v2 — System / Global Hotkeys
=========================================
Process-wide hotkey listener. The user can hit F7/F8/F9 with Forza focused
and have something happen even though our Chromium window isn't focused.

Module 1 starts the listener but registers no handlers. Later modules call
`register(key, fn)` to attach behavior:

    Module 8 (drive):  register("F8", drive_panic)
    Module 8 (ppo):    register("F7", ppo_toggle)
    Module 8 (drive):  register("F9", gamepad_panic)

We deliberately don't import drive/ppo from here — those modules register
themselves at boot. This keeps the dependency graph clean.

Implementation
--------------
We use the `keyboard` library on Windows. Because handlers may want to
mutate global state in *other* modules, we don't call them directly from
the keyboard thread (callbacks there should be fast and non-blocking).
Instead, every handler is dispatched as an HTTP POST to the FastAPI side
server, which runs in a normal request thread. This keeps the keyboard
listener thread responsive and means hotkey behavior can be tested with
plain `curl`.

If the `keyboard` library is unavailable (e.g. on Linux without root, or in
a CI environment), `start()` becomes a no-op and logs a warning. The app
still boots — hotkeys are a nice-to-have, not required.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

log = logging.getLogger("forzatek.hotkeys")

# (key_str, handler_fn). Handlers run on a worker thread.
_HANDLERS: dict[str, Callable[[], None]] = {}
_started = False
_lock = threading.Lock()


def register(key: str, handler: Callable[[], None]) -> None:
    """Attach a handler to a key. Replaces any existing handler for that key.

    `key` uses `keyboard` library syntax: 'f7', 'f8', 'ctrl+shift+p', etc.
    """
    with _lock:
        _HANDLERS[key.lower()] = handler
        log.info("registered handler for %s", key.lower())


def unregister(key: str) -> None:
    with _lock:
        _HANDLERS.pop(key.lower(), None)


def _dispatch(key: str) -> None:
    """Run a handler off the keyboard listener thread."""
    handler = _HANDLERS.get(key)
    if handler is None:
        return

    def _run():
        try:
            handler()
        except Exception as e:
            log.error("handler for %s crashed: %s", key, e)

    threading.Thread(target=_run, daemon=True, name=f"hotkey-{key}").start()


def start() -> None:
    """Bind global hotkeys. Idempotent.

    Raises RuntimeError if the `keyboard` library can't be imported. Callers
    in `main.py` catch this and treat it as non-fatal.
    """
    global _started
    if _started:
        return

    try:
        import keyboard  # type: ignore
    except ImportError as e:
        raise RuntimeError(f"keyboard library not available: {e}") from e

    # We don't bind specific keys eagerly — we listen to ALL key-down events
    # and look them up in our handler map. This way `register()` can be
    # called after `start()` and the new handler takes effect immediately,
    # without having to rebind keys.
    def _on_event(event):
        if event.event_type != "down":
            return
        # Build the canonical key string. For now we only handle plain keys;
        # modifier combos can be added later with `keyboard.is_pressed()`.
        name = (event.name or "").lower()
        if name and name in _HANDLERS:
            _dispatch(name)

    try:
        keyboard.hook(_on_event)
    except Exception as e:
        # Common cause on Linux: needs root. We treat as non-fatal.
        raise RuntimeError(f"could not hook keyboard: {e}") from e

    _started = True
    log.info("global hotkey listener started")


def stop() -> None:
    """Currently a no-op — `keyboard` hooks live for the process lifetime
    and are cleaned up on exit. Reserved for future use."""
    global _started
    _started = False