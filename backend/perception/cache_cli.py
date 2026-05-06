"""
Module 5 — Cache CLI
====================
Spawned by runner.start_cache_build() so the cache build can run in its own
subprocess (won't take the Eel app down if it crashes, won't fight with
training for the same Python interpreter state).

Usage:
    python -m backend.perception.cache_cli --build
    python -m backend.perception.cache_cli --clear
    python -m backend.perception.cache_cli --status
"""
from __future__ import annotations

import argparse
import json
import sys

from backend.perception import cache as _cache


def main() -> int:
    p = argparse.ArgumentParser(description="Build / inspect the perception cache.")
    p.add_argument("--build",  action="store_true", help="Build (or rebuild) the cache.")
    p.add_argument("--clear",  action="store_true", help="Delete the cache files.")
    p.add_argument("--status", action="store_true", help="Print status JSON and exit.")
    args = p.parse_args()

    if args.status:
        print(json.dumps(_cache.status(), indent=2))
        return 0
    if args.clear:
        ok = _cache.clear()
        print(json.dumps({"cleared": bool(ok)}))
        return 0
    if args.build:
        result = _cache.build()
        return 0 if result.get("ok") else 1

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())