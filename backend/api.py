from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import os
import json
import io
from typing import Optional, List
import base64

from .episodic_engine import (
    generate_episodic_intelligence,
    generate_character_development,
    generate_dialogue_suggestions,
    generate_visual_mood_board,
    generate_music_recommendations,
    generate_shot_composition,
)

try:
    from google.cloud import texttospeech
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class AnalyseRequest(BaseModel):
    core_idea: str = Field(..., description="Creator's core vertical series idea")
    episode_count: int = Field(
        6, ge=5, le=8, description="Number of episodes (5–8 inclusive)"
    )
    model_name: str = Field(
        "gemini-2.5-flash",
        description="Gemini model identifier",
    )
    genre: Optional[str] = Field("drama", description="Story genre")
    tone: Optional[str] = Field("dramatic", description="Narrative tone")
    target_audience: Optional[str] = Field("18-35", description="Target audience")


class CharacterRequest(BaseModel):
    core_idea: str
    episode_count: int = 6
    model_name: str = "gemini-2.5-flash"


class DialogueRequest(BaseModel):
    episode_data: dict
    model_name: str = "gemini-2.5-flash"


class MoodBoardRequest(BaseModel):
    episode_data: dict
    model_name: str = "gemini-2.5-flash"


class MusicRequest(BaseModel):
    emotional_arcs: List[dict]
    model_name: str = "gemini-2.5-flash"


class ShotCompositionRequest(BaseModel):
    episode_data: dict
    model_name: str = "gemini-2.5-flash"


class TextToSpeechRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    voice_name: str = Field("en-US-Neural2-C", description="Google Cloud voice")
    language_code: str = Field("en-US", description="Language code")


app = FastAPI(
    title="VBOX Episodic Intelligence Engine API",
    description="AI-powered narrative intelligence for vertical episodic storytelling.",
    version="2.0.0",
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
    """Generate complete episodic analysis with all features"""
    if not payload.core_idea.strip():
        raise HTTPException(status_code=400, detail="core_idea cannot be empty.")

    try:
        # Generate main episodic intelligence
        result = generate_episodic_intelligence(
            core_idea=payload.core_idea,
            desired_episodes=payload.episode_count,
            model_name=payload.model_name,
            genre=payload.genre,
            tone=payload.tone,
            target_audience=payload.target_audience,
        )
        
        # Generate character development
        character_data = await generate_character_development(
            core_idea=payload.core_idea,
            episodes=payload.episode_count,
            model_name=payload.model_name,
        )
        
        result["character_development"] = character_data
        
        # Generate dialogue suggestions for first episode
        if "episodes" in result and len(result["episodes"]) > 0:
            dialogue_data = await generate_dialogue_suggestions(
                episode=result["episodes"][0],
                model_name=payload.model_name,
            )
            result["sample_dialogues"] = dialogue_data
        
        # Generate visual mood board
        mood_board = await generate_visual_mood_board(
            core_idea=payload.core_idea,
            genre=payload.genre,
            model_name=payload.model_name,
        )
        result["visual_mood_board"] = mood_board
        
        # Generate music recommendations
        emotional_arcs = []
        if "episodes" in result:
            for ep in result["episodes"]:
                if "emotional_arc_analysis" in ep:
                    emotional_arcs.extend(ep["emotional_arc_analysis"])
        
        music_recs = await generate_music_recommendations(
            emotional_arcs=emotional_arcs,
            model_name=payload.model_name,
        )
        result["music_recommendations"] = music_recs
        
        # Generate shot composition suggestions
        if "episodes" in result and len(result["episodes"]) > 0:
            shot_recs = await generate_shot_composition(
                episode=result["episodes"][0],
                model_name=payload.model_name,
            )
            result["shot_composition"] = shot_recs
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/characters")
async def get_characters(payload: CharacterRequest):
    """Generate detailed character development"""
    try:
        result = await generate_character_development(
            core_idea=payload.core_idea,
            episodes=payload.episode_count,
            model_name=payload.model_name,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/dialogue")
async def get_dialogue(payload: DialogueRequest):
    """Generate dialogue suggestions"""
    try:
        result = await generate_dialogue_suggestions(
            episode=payload.episode_data,
            model_name=payload.model_name,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/mood-board")
async def get_mood_board(payload: MoodBoardRequest):
    """Generate visual mood board recommendations"""
    try:
        result = await generate_visual_mood_board(
            core_idea=payload.episode_data.get("series_title", ""),
            genre="drama",
            model_name=payload.model_name,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/music")
async def get_music(payload: MusicRequest):
    """Generate music recommendations based on emotional arcs"""
    try:
        result = await generate_music_recommendations(
            emotional_arcs=payload.emotional_arcs,
            model_name=payload.model_name,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/shots")
async def get_shot_composition(payload: ShotCompositionRequest):
    """Generate shot composition suggestions"""
    try:
        result = await generate_shot_composition(
            episode=payload.episode_data,
            model_name=payload.model_name,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/tts")
async def text_to_speech(payload: TextToSpeechRequest):
    """Convert text to speech using Google Cloud TTS"""
    if not TTS_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail="Text-to-speech not available. Set up Google Cloud credentials."
        )
    
    try:
        client = texttospeech.TextToSpeechClient()
        
        input_text = texttospeech.SynthesisInput(text=payload.text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=payload.language_code,
            name=payload.voice_name,
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )
        
        audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
        
        return {
            "audio": audio_base64,
            "mime_type": "audio/mpeg",
            "duration_ms": len(response.audio_content) // 128,  # Approximate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


