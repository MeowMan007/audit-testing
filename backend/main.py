"""
AccessLens — AI-Powered Accessibility Audit Tool
FastAPI Application Entry Point

This is the main application file that:
1. Creates the FastAPI app instance
2. Configures CORS for frontend access
3. Mounts the frontend static files
4. Includes the audit API router
5. Provides a root redirect to the frontend

Usage:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

# Create FastAPI app
app = FastAPI(
    title="AccessLens — AI Accessibility Audit Tool",
    description=(
        "AI-powered web accessibility auditing tool that combines "
        "WCAG 2.1 rule-based checks with a custom-trained EfficientNet-B0 CNN "
        "model for comprehensive accessibility analysis."
    ),
    version="1.0.0",
)

# CORS — allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
from backend.routers.audit import router as audit_router
app.include_router(audit_router)

# Mount frontend static files
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def root():
        """Redirect root to the frontend."""
        return RedirectResponse(url="/static/index.html")
else:
    @app.get("/")
    async def root():
        return {"message": "AccessLens API is running. Frontend not found — serve frontend/index.html separately."}


@app.on_event("shutdown")
async def shutdown():
    """Clean up resources on shutdown."""
    from backend.routers.audit import page_fetcher
    page_fetcher.close()
