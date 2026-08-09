"""Uvicorn entrypoint for the local Stage 2 API."""

from rural_stroke_assist.server.app import create_app

app = create_app()
