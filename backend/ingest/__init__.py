"""ForzaTek AI v2 — Module 2: Ingest.

Pure producer module. The only code in the project that writes to the
`frames` table. Three sources converge here:
    1. Live game capture       -> recorder.py
    2. Local video file walk   -> video_ingester.py
    3. YouTube download + walk -> video_ingester.py (via yt-dlp)

Public surface (called from backend/main.py):
    from backend.ingest import eel_api, routes
    eel_api.register_eel(eel)
    routes.register_routes(app)
"""