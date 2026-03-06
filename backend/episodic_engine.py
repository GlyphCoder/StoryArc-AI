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


def _build_user_prompt(core_idea: str, desired_episodes: int) -> str:
    return f"""
Core story idea provided by the creator:

\"\"\"{core_idea}\"\"\"

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
) -> Dict[str, Any]:
    """Single entry point used by API and Streamlit."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    model = _get_model(model_name)
    prompt = _build_user_prompt(core_idea, desired_episodes)
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


