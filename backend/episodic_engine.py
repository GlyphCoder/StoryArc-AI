import os
import json
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


EPISODIC_SYSTEM_PROMPT = """
You are the 'Episodic Intelligence Engine,' an advanced AI designed to optimize vertical, short-form storytelling (90-second format).

Take the user's core story idea and decompose it into a highly engaging, retention-optimized vertical series (5 to 8 episodes).

You MUST output the result strictly as a JSON object that conforms to this schema (no surrounding explanation, no Markdown):

{
  "series_title": "string",
  "series_logline": "string",
  "total_episodes": "integer",
  "episodes": [
    {
      "episode_number": "integer",
      "episode_title": "string",
      "narrative_breakdown": "string",
      "emotional_arc_analysis": [
        {
          "time_block": "string",
          "dominant_emotion": "string",
          "engagement_level": "string",
          "flat_zone_warning": "boolean"
        }
      ],
      "cliffhanger_scoring": {
        "description": "string",
        "strength_score": "integer",
        "explanation": "string"
      },
      "retention_risk_prediction": [
        {
          "drop_off_timestamp": "string",
          "risk_level": "string",
          "reason_for_dropoff": "string"
        }
      ],
      "optimisation_suggestions": [
        "string"
      ]
    }
  ]
}

Rules:
- "total_episodes" must be between 5 and 8 inclusive.
- All timestamps must assume a single 90-second vertical video per episode.
- Do NOT include any keys beyond this schema.
- Do NOT wrap JSON in backticks.
- Do NOT include any natural language commentary outside the JSON.
"""


def _get_model(model_name: str) -> genai.GenerativeModel:
    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=EPISODIC_SYSTEM_PROMPT,
        generation_config={
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 40,
            "response_mime_type": "application/json",
        },
    )


def _build_user_prompt(core_idea: str, desired_episodes: int, genre: str = "", tone: str = "", target_audience: str = "") -> str:
    extra_context = ""
    if genre:
        extra_context += f"\nGenre: {genre}"
    if tone:
        extra_context += f"\nTone: {tone}"
    if target_audience:
        extra_context += f"\nTarget Audience: {target_audience}"
    
    return f"""
Core story idea provided by the creator:

\"\"\"{core_idea}\"\"\"{extra_context}

Instructions:
- Decompose this idea into a vertical series optimized for short-form platforms.
- You MUST create exactly {desired_episodes} episodes (even though the global rule is 5–8).
- Assume each episode is a single 90-second vertical video.
- Respect the schema given in the system prompt.
- Output ONLY the final JSON object, no commentary or Markdown.
"""


def _safe_parse_json(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None

    candidate = raw_text.strip()
    if "{" in candidate and "}" in candidate:
        start = candidate.find("{")
        end = candidate.rfind("}") + 1
        candidate = candidate[start:end]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _compute_engine_metrics(series_json: Dict[str, Any]) -> Dict[str, Any]:
    """Heuristic metrics on top of LLM output for hackathon scoring."""
    episodes: List[Dict[str, Any]] = series_json.get("episodes", []) or []

    cliff_scores: List[int] = []
    high_risk_segments = 0
    medium_risk_segments = 0
    flat_zone_warnings = 0

    episode_flatness_index: List[Tuple[int, float]] = []

    for ep in episodes:
        ep_no = int(ep.get("episode_number", 0) or 0)
        emotional_arc = ep.get("emotional_arc_analysis", []) or []
        cliff = (ep.get("cliffhanger_scoring") or {}).get("strength_score")

        if isinstance(cliff, int):
            cliff_scores.append(cliff)

        # Count risk segments
        for risk in ep.get("retention_risk_prediction", []) or []:
            level = str(risk.get("risk_level", "")).lower()
            if "high" in level:
                high_risk_segments += 1
            elif "medium" in level:
                medium_risk_segments += 1

        # Compute a "flatness index" per episode: ratio of flat zones to arc points
        flat_blocks = 0
        for arc_point in emotional_arc:
            if bool(arc_point.get("flat_zone_warning", False)):
                flat_blocks += 1
                flat_zone_warnings += 1

        total_blocks = max(len(emotional_arc), 1)
        episode_flatness_index.append((ep_no, flat_blocks / total_blocks))

    avg_cliff = sum(cliff_scores) / len(cliff_scores) if cliff_scores else 0.0
    worst_flat = max(episode_flatness_index, key=lambda x: x[1])[0] if episode_flatness_index else None

    return {
        "avg_cliffhanger_strength": round(avg_cliff, 2),
        "high_risk_segments": high_risk_segments,
        "medium_risk_segments": medium_risk_segments,
        "flat_zone_warnings": flat_zone_warnings,
        "flattest_episode": worst_flat,
    }


def generate_episodic_intelligence(
    core_idea: str,
    desired_episodes: int,
    model_name: str = "gemini-2.5-flash",
    genre: str = "",
    tone: str = "",
    target_audience: str = "",
) -> Dict[str, Any]:
    """Single entry point used by API and Streamlit."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    model = _get_model(model_name)
    prompt = _build_user_prompt(core_idea, desired_episodes, genre, tone, target_audience)
    response = model.generate_content(prompt)

    raw_text = getattr(response, "text", None)
    parsed = _safe_parse_json(raw_text)
    if not parsed:
        raise ValueError("Failed to parse JSON from Gemini response.")

    engine_metrics = _compute_engine_metrics(parsed)

    return {
        "series": parsed,
        "engine_metrics": engine_metrics,
        "raw_text": raw_text,
    }


# ============================================================================
# NEW FEATURE ENGINES (Advanced Capabilities)
# ============================================================================

async def generate_character_development(
    core_idea: str,
    episodes: int,
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """Generate detailed character development arcs and profiles"""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    
    model = genai.GenerativeModel(model_name=model_name)
    
    prompt = f"""
    Based on this story concept: "{core_idea}"
    
    Create detailed character profiles for a {episodes}-episode vertical series.
    For each character, provide:
    1. Character name and archetype
    2. Background/motivation
    3. Arc across the series
    4. Key personality traits
    5. Visual appearance suggestions
    
    Format as JSON with structure:
    {{
        "characters": [
            {{
                "name": "string",
                "archetype": "string",
                "background": "string",
                "character_arc": "string",
                "traits": ["string"],
                "visual_description": "string",
                "appearance_evolution": ["string"]
            }}
        ],
        "ensemble_dynamics": "string"
    }}
    
    Output ONLY valid JSON, no markdown or explanations.
    """
    
    response = model.generate_content(prompt)
    raw_text = getattr(response, "text", None)
    parsed = _safe_parse_json(raw_text)
    
    return parsed or {"characters": [], "ensemble_dynamics": ""}


async def generate_dialogue_suggestions(
    episode: Dict[str, Any],
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """Generate realistic dialogue suggestions for episodes"""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    
    model = genai.GenerativeModel(model_name=model_name)
    
    ep_title = episode.get("episode_title", "Episode")
    narrative = episode.get("narrative_breakdown", "")
    
    prompt = f"""
    Create realistic dialogue snippets for this episode:
    Title: {ep_title}
    Narrative: {narrative}
    
    Provide natural, engaging dialogue that fits the 90-second format.
    Include:
    1. Opening hook dialogue (0-10 seconds)
    2. Mid-point tension dialogue (30-60 seconds)
    3. Cliffhanger dialogue (75-90 seconds)
    
    Each dialogue should be short, punchy, and engaging.
    Format as JSON:
    {{
        "dialogue_groups": [
            {{
                "time_block": "string",
                "speaker": "string (or 'Multiple')",
                "lines": ["string"],
                "emotion": "string",
                "purpose": "string"
            }}
        ]
    }}
    
    Output ONLY valid JSON, no markdown.
    """
    
    response = model.generate_content(prompt)
    raw_text = getattr(response, "text", None)
    parsed = _safe_parse_json(raw_text)
    
    return parsed or {"dialogue_groups": []}


async def generate_visual_mood_board(
    core_idea: str,
    genre: str = "drama",
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """Generate visual mood board recommendations"""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    
    model = genai.GenerativeModel(model_name=model_name)
    
    prompt = f"""
    Create a comprehensive visual mood board guide for a vertical video series:
    Concept: {core_idea}
    Genre: {genre}
    
    Provide specific visual recommendations:
    1. Color palette (include hex codes)
    2. Camera filters/editing style
    3. Lighting design
    4. Location aesthetics
    5. Props and set design
    6. Typography/text overlays
    7. Transition styles
    8. Animation/VFX suggestions
    
    Format as JSON:
    {{
        "color_palette": {{
            "primary": ["string (hex)"],
            "secondary": ["string (hex)"],
            "accent": ["string (hex)"],
            "reasoning": "string"
        }},
        "camera_style": "string",
        "lighting_design": "string",
        "locations": ["string"],
        "props_and_sets": ["string"],
        "typography": "string",
        "transitions": ["string"],
        "vfx_suggestions": ["string"]
    }}
    
    Output ONLY valid JSON.
    """
    
    response = model.generate_content(prompt)
    raw_text = getattr(response, "text", None)
    parsed = _safe_parse_json(raw_text)
    
    return parsed or {"color_palette": {}, "camera_style": ""}


async def generate_music_recommendations(
    emotional_arcs: List[Dict[str, Any]],
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """Generate music and sound design recommendations"""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    
    model = genai.GenerativeModel(model_name=model_name)
    
    emotions_summary = str(emotional_arcs)[:500]  # Truncate for API
    
    prompt = f"""
    Create music and sound design recommendations based on these emotional arcs:
    {emotions_summary}
    
    Provide specific music suggestions:
    1. Opening theme (genre, mood, tempo)
    2. Background music by emotion
    3. Sound effects recommendations
    4. Silence/breathing room placement
    5. Music swells for climactic moments
    6. Ending theme
    
    Include reference genres/artists where helpful.
    
    Format as JSON:
    {{
        "opening_theme": {{
            "genre": "string",
            "mood": "string",
            "tempo_bpm": "integer",
            "style": "string"
        }},
        "background_music": [
            {{
                "emotion": "string",
                "genre": "string",
                "reference_artists": ["string"],
                "duration": "string"
            }}
        ],
        "sound_effects": ["string"],
        "silence_placement": ["string"],
        "climactic_swells": ["string"],
        "ending_theme": "string"
    }}
    
    Output ONLY valid JSON.
    """
    
    response = model.generate_content(prompt)
    raw_text = getattr(response, "text", None)
    parsed = _safe_parse_json(raw_text)
    
    return parsed or {"opening_theme": {}, "background_music": []}


async def generate_shot_composition(
    episode: Dict[str, Any],
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """Generate shot composition and framing suggestions"""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    
    model = genai.GenerativeModel(model_name=model_name)
    
    ep_title = episode.get("episode_title", "")
    narrative = episode.get("narrative_breakdown", "")
    
    prompt = f"""
    Create detailed shot composition guides for this episode of a vertical video:
    Title: {ep_title}
    Story: {narrative}
    
    Break down shots by time blocks:
    1. Opening shot (0-15s): Hook the viewer
    2. Build shots (15-60s): Develop the story
    3. Climax shot (60-75s): Peak moment
    4. Cliffhanger shot (75-90s): Leave them wanting more
    
    For each section provide:
    - Shot type (wide, medium, close-up, etc.)
    - Camera movement (pan, zoom, static)
    - Subject framing
    - Depth of field
    - Focus pulling
    
    Format as JSON:
    {{
        "shot_breakdown": [
            {{
                "time_block": "string",
                "shot_type": "string",
                "camera_movement": "string",
                "framing": "string",
                "depth_of_field": "string",
                "focus_pulling": "string",
                "purpose": "string"
            }}
        ],
        "technical_notes": "string",
        "equipment_suggestions": ["string"]
    }}
    
    Output ONLY valid JSON.
    """
    
    response = model.generate_content(prompt)
    raw_text = getattr(response, "text", None)
    parsed = _safe_parse_json(raw_text)
    
    return parsed or {"shot_breakdown": [], "technical_notes": ""}

