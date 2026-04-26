"""
ForzaTek AI v2 — Module 2 / FastAPI Routes
============================================
Only the live MJPEG stream of the recorder lives here. Everything else
goes through Eel (see eel_api.py).

Why MJPEG and not Eel?
----------------------
Eel's bridge is JSON-only — pushing a 1280x720 JPEG frame over it 15
times a second would be slow and would block the bridge for everything
else. Browsers render `multipart/x-mixed-replace` natively into an
`<img>` tag, so we hand the bytes back through FastAPI's
`StreamingResponse` and wire the page to `<img src="/api/ingest/preview.mjpg">`.
"""
from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, Response

from backend.ingest.recorder import RECORDER

log = logging.getLogger("forzatek.ingest.routes")

_BOUNDARY = "frame"


def register_routes(app: FastAPI) -> None:
    """Called once from `backend/main.py`."""

    @app.get("/api/ingest/preview.mjpg")
    async def preview_mjpg():
        """Endless MJPEG stream of the recorder's latest frame.

        Pulls `RECORDER.latest_jpeg()` at ~10 Hz; if no frame is available
        yet (recorder idle), serves a small placeholder JPEG."""
        return StreamingResponse(
            _mjpeg_iter(),
            media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
        )

    @app.get("/api/ingest/snapshot.jpg")
    async def preview_snapshot():
        """Single JPEG of the latest frame (for thumbnails / non-streaming UIs)."""
        jpeg = RECORDER.latest_jpeg()
        if jpeg is None:
            return Response(
                content=_PLACEHOLDER_JPEG,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/ingest/health")
    async def ingest_health():
        return JSONResponse({"ok": True, "ts": time.time()})

    log.info(
        "ingest: registered routes "
        "[/api/ingest/preview.mjpg, /api/ingest/snapshot.jpg, /api/ingest/health]"
    )


# ─── MJPEG iterator ────────────────────────────────────────────────────────

async def _mjpeg_iter():
    """Async generator that yields multipart MJPEG chunks."""
    last_sent = 0
    placeholder_sent = False
    while True:
        jpeg = RECORDER.latest_jpeg()
        now = time.time()

        if jpeg is None:
            # Send the placeholder once, then idle so we don't spam the
            # browser with the same gray frame.
            if not placeholder_sent:
                yield _multipart_chunk(_PLACEHOLDER_JPEG)
                placeholder_sent = True
            await asyncio.sleep(0.5)
            continue

        placeholder_sent = False
        # Throttle to ~10 Hz so MJPEG-decode in the browser stays cheap;
        # the recorder may run faster, that's fine.
        if now - last_sent >= 0.10:
            yield _multipart_chunk(jpeg)
            last_sent = now
        await asyncio.sleep(0.02)


def _multipart_chunk(jpeg: bytes) -> bytes:
    head = (
        f"--{_BOUNDARY}\r\n"
        f"Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(jpeg)}\r\n\r\n"
    ).encode("ascii")
    return head + jpeg + b"\r\n"


# ─── Tiny placeholder JPEG (128x72 dark gray) ─────────────────────────────
# Generated once at import time via numpy + cv2 if available; otherwise a
# baked-in 1x1 gray JPEG.

def _build_placeholder() -> bytes:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        img = np.full((72, 128, 3), 24, dtype=np.uint8)
        cv2.putText(
            img, "no signal", (16, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA,
        )
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    # Smallest valid JPEG (1x1 gray).
    return bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707"
        "07090908'a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c"
        "1c2837292c30313434341f27393d38323c2e333432ffc00011080001000103012200"
        "021101031101ffc4001f0000010501010101010100000000000000000102030405"
        "060708090a0bffc400b5100002010303020403050504040000017d010203000411"
        "05122131410613516107227114328191a1082342b1c11552d1f02433627282090a"
        "161718191a25262728292a3435363738393a434445464748494a53545556575859"
        "5a636465666768696a737475767778797a838485868788898a92939495969798"
        "999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4"
        "d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffc4001f0100"
        "030101010101010101010000000000000102030405060708090a0bffc400b5"
        "11000201020404030407050404000102770001020311040521310612415107"
        "61711322328108144291a1b1c109233352f0156272d10a162434e125f1171"
        "8191a262728292a35363738393a434445464748494a535455565758595a63"
        "6465666768696a737475767778797a82838485868788898a92939495969798"
        "999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2"
        "d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda000c"
        "03010002110311003f00fbfcaffd9".replace("'", "")
    )


_PLACEHOLDER_JPEG = _build_placeholder()


__all__ = ["register_routes"]
