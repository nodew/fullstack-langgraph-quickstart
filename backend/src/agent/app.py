# mypy: disable - error - code = "no-untyped-def,misc"
import pathlib
import os
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import fastapi.exceptions
from typing import Dict, List, Any
from agent.models import get_supported_models

# Define the FastAPI app
app = FastAPI()

@app.get("/api/providers")
async def get_available_models() -> List[Dict[str, Any]]:
    """
    Get available LLM models based on environment configuration.
    Returns models for providers that have their API keys configured.
    """
    available_providers = []
    supported_models = get_supported_models()

    # Check which providers have API keys configured
    provider_env_vars = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "azure_openai": ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
    }

    for provider, env_vars in provider_env_vars.items():
        if isinstance(env_vars, list):
            # For Azure OpenAI, check that both endpoint and API key are set
            if all(os.getenv(var) for var in env_vars):
                available_providers.append({
                    "provider": provider,
                    "models": supported_models.get(provider, [])
                })
        else:
            # For other providers, check single API key
            if os.getenv(env_vars):
                available_providers.append({
                    "provider": provider,
                    "models": supported_models.get(provider, [])
                })

    return available_providers

def create_frontend_router(build_dir="../frontend/dist"):
    """Creates a router to serve the React frontend.

    Args:
        build_dir: Path to the React build directory relative to this file.

    Returns:
        A Starlette application serving the frontend.
    """
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir
    static_files_path = build_path / "assets"  # Vite uses 'assets' subdir

    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        print(
            f"WARN: Frontend build directory not found or incomplete at {build_path}. Serving frontend will likely fail."
        )
        # Return a dummy router if build isn't ready
        from starlette.routing import Route

        async def dummy_frontend(request):
            return Response(
                "Frontend not built. Run 'npm run build' in the frontend directory.",
                media_type="text/plain",
                status_code=503,
            )

        return Route("/{path:path}", endpoint=dummy_frontend)

    build_dir = pathlib.Path(build_dir)

    react = FastAPI(openapi_url="")
    react.mount(
        "/assets", StaticFiles(directory=static_files_path), name="static_assets"
    )

    @react.get("/{path:path}")
    async def handle_catch_all(request: Request, path: str):
        fp = build_path / path
        if not fp.exists() or not fp.is_file():
            fp = build_path / "index.html"
        return fastapi.responses.FileResponse(fp)

    return react


# Mount the frontend under /app to not conflict with the LangGraph API routes
app.mount(
    "/app",
    create_frontend_router(),
    name="frontend",
)
