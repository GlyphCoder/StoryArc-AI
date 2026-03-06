from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.episodic_engine import generate_episodic_intelligence


class AnalyseRequest(BaseModel):
    core_idea: str = Field(..., description="Creator's core vertical series idea")
    episode_count: int = Field(
        6, ge=5, le=8, description="Number of episodes (5–8 inclusive)"
    )
    model_name: str = Field(
        "gemini-2.5-flash",
        description="Gemini model identifier",
    )


app = FastAPI(
    title="VBOX Episodic Intelligence Engine API",
    description="AI-powered narrative intelligence for vertical episodic storytelling.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/analyse")
async def analyse_story(payload: AnalyseRequest):
    if not payload.core_idea.strip():
        raise HTTPException(status_code=400, detail="core_idea cannot be empty.")

    try:
        result = generate_episodic_intelligence(
            core_idea=payload.core_idea,
            desired_episodes=payload.episode_count,
            model_name=payload.model_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return result


@app.get("/health")
async def health_check():
    return {"status": "ok"}


