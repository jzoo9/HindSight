"""
Vercel entrypoint.

Vercel auto-detects a FastAPI instance named `app` in `app.py`.
"""

from api.main import app  # re-export for Vercel

