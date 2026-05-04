from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.shared.config import get_settings

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"NexusAI API v{settings.app_version} starting...")
    print(f"ENV: {settings.env}")
    print(f"LLM model: {settings.llm_model}")
    yield
    print("NexusAI API shutting down...")

app = FastAPI(
    title="NexusAI API",
    version=settings.app_version,
    description="Academic AI assistant backend for Moodle",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": settings.app_version,
        "env": settings.env,
        "llm_model": settings.llm_model,
    }

@app.get("/")
async def root():
    return {"message": "NexusAI API running"}
