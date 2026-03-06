import os
import json
from typing import Any, Dict, List, Optional

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd

# ---------------------------------------------------------
# Config & Initialization
# ---------------------------------------------------------

# Load local .env if present (useful for hackathon dev)
load_dotenv()

# Read Gemini API key from environment
# Expecting GEMINI_API_KEY to be defined in a .env file or shell env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Google Generative AI client if key is available
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Core system prompt embedded as required
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


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_model(model_name: str) -> genai.GenerativeModel:
    """
    Create and return a configured GenerativeModel instance.
    Using response_mime_type='application/json' to strongly bias the model to JSON output.
    """
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


def build_user_prompt(core_idea: str, desired_episodes: int) -> str:
    """
    Build the user-facing prompt that:
    - Injects the creator's story idea.
    - Reinforces episode count and JSON-only requirements.
    """
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


def safe_parse_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON from the model output.
    - Strips whitespace.
    - Handles stray leading/trailing text defensively.
    """
    if not raw_text:
        return None

    candidate = raw_text.strip()

    # In case the model accidentally wraps JSON with text, try to locate the first '{' and last '}'.
    if "{" in candidate and "}" in candidate:
        start = candidate.find("{")
        end = candidate.rfind("}") + 1
        candidate = candidate[start:end]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def compute_high_level_metrics(series_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive some aggregate metrics for dashboard-style visualization:
    - Average cliffhanger strength across episodes.
    - Count of episodes with any 'high' retention risk.
    - Count of total flat-zone warnings.
    """
    episodes: List[Dict[str, Any]] = series_json.get("episodes", []) or []

    cliff_scores: List[int] = []
    high_risk_count = 0
    flat_zone_count = 0

    for ep in episodes:
        cliff = (ep.get("cliffhanger_scoring") or {}).get("strength_score")
        if isinstance(cliff, int):
            cliff_scores.append(cliff)

        for risk in ep.get("retention_risk_prediction", []) or []:
            risk_level = str(risk.get("risk_level", "")).lower()
            if "high" in risk_level:
                high_risk_count += 1

        for arc_point in ep.get("emotional_arc_analysis", []) or []:
            if bool(arc_point.get("flat_zone_warning", False)):
                flat_zone_count += 1

    avg_cliff = sum(cliff_scores) / len(cliff_scores) if cliff_scores else 0.0

    return {
        "avg_cliffhanger_strength": round(avg_cliff, 2),
        "episodes_with_high_risk_segments": high_risk_count,
        "flat_zone_warnings": flat_zone_count,
    }


def episodes_to_dataframe(episodes: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert episode list to a compact DataFrame for quick overview:
    - Maps episode number, title, and overall cliffhanger score.
    """
    rows = []
    for ep in episodes:
        rows.append(
            {
                "Episode #": ep.get("episode_number"),
                "Title": ep.get("episode_title"),
                "Cliffhanger Score": (ep.get("cliffhanger_scoring") or {}).get(
                    "strength_score"
                ),
            }
        )
    return pd.DataFrame(rows)


def emotional_arc_to_dataframe(emotional_arc: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert emotional arc list into a DataFrame for table display.
    """
    if not emotional_arc:
        return pd.DataFrame()
    return pd.DataFrame(emotional_arc)


def retention_risk_to_dataframe(risks: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert retention risk list into a DataFrame for table display.
    """
    if not risks:
        return pd.DataFrame()
    return pd.DataFrame(risks)


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

# Set page-wide configuration for a more "product-like" feel
st.set_page_config(
    page_title="VBOX Episodic Intelligence Engine",
    page_icon="🎬",
    layout="wide",
)

# Top-level layout: header + description
st.title("VBOX Episodic Intelligence Engine")
st.caption(
    "AI-powered narrative intelligence for 90-second vertical series • Built for the Quantloop AI + Creator Economy Hackathon."
)

# Sidebar: configuration panel for creators and judges
with st.sidebar:
    st.header("⚙️ Configuration")

    if not GEMINI_API_KEY:
        st.error(
            "GEMINI_API_KEY not found.\n\n"
            "Set it as an environment variable or in a local `.env` file."
        )

    # Allow model selection (restricted to a known-working model)
    model_name = st.selectbox(
        "Gemini model",
        options=["gemini-2.5-flash"],
        index=0,
        help="Using gemini-2.5-flash for reliability.",
    )

    # Allow creators/judges to lock in an episode count within the allowed band
    desired_episodes = st.slider(
        "Number of episodes",
        min_value=5,
        max_value=8,
        value=6,
        help="Episodic decomposition between 5 and 8 parts.",
    )

    st.markdown("---")
    st.markdown(
        "This engine analyses:\n"
        "- Story decomposition\n"
        "- Emotional arc per time block\n"
        "- Cliffhanger strength\n"
        "- Retention risk (drop-offs)\n"
        "- Optimisation suggestions\n"
    )

# Main panel: story input + results
with st.container():
    st.subheader("1️⃣ Input your core story idea")

    core_idea = st.text_area(
        "Describe your vertical series concept",
        height=200,
        placeholder=(
            "Example: A rookie urban magician tries to expose a viral fake psychic cult, "
            "only to discover their illusions are powered by real, dangerous tech..."
        ),
    )

    col_run, col_info = st.columns([1, 2])
    with col_run:
        run_button = st.button(
            "Generate Episodic Intelligence",
            type="primary",
            use_container_width=True,
        )
    with col_info:
        st.markdown(
            "<small>The engine will decompose your idea into a 5–8 part vertical series "
            "and run multi-layer narrative analytics.</small>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# Inference & Display Logic
# ---------------------------------------------------------

if run_button:
    if not GEMINI_API_KEY:
        st.stop()

    if not core_idea.strip():
        st.warning("Please provide a core story idea before running the engine.")
        st.stop()

    # Wrap the full pipeline in a spinner so judges see responsiveness
    with st.spinner("Calling Episodic Intelligence Engine..."):
        try:
            model = get_model(model_name)
            user_prompt = build_user_prompt(core_idea, desired_episodes)

            # Call Gemini with our system prompt + user prompt
            response = model.generate_content(user_prompt)

            # Extract text (JSON string) from the response object
            raw_text = getattr(response, "text", None)
            parsed = safe_parse_json(raw_text)

            if not parsed:
                # Show developer-level diagnostics for hackathon debugging
                st.error("Failed to parse JSON from Gemini response.")
                with st.expander("Raw model output (for debugging)", expanded=False):
                    st.code(raw_text or "No text returned.", language="json")
                st.stop()

        except Exception as e:
            # Generic guardrail: never crash the Streamlit app
            st.error(f"Unexpected error while calling Gemini: {e}")
            st.stop()

    # -----------------------------------------------------
    # High-level Series Overview (top of the page)
    # -----------------------------------------------------
    st.success("Episodic Intelligence successfully generated.")

    series_title = parsed.get("series_title", "Untitled Series")
    series_logline = parsed.get("series_logline", "")
    total_episodes = parsed.get("total_episodes", 0)
    episodes = parsed.get("episodes", []) or []

    st.subheader("2️⃣ Series Overview")
    st.markdown(f"### 🎥 {series_title}")
    if series_logline:
        st.markdown(f"**Logline:** {series_logline}")

    metrics = compute_high_level_metrics(parsed)

    # Use three metric cards for instant readability
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Episodes", total_episodes)
    m2.metric("Avg. Cliffhanger Strength (1–10)", metrics["avg_cliffhanger_strength"])
    m3.metric("Episodes with High-Risk Segments", metrics["episodes_with_high_risk_segments"])
    m4.metric("Flat-Zone Warnings", metrics["flat_zone_warnings"])

    # Compact table of episodes for quick scan
    if episodes:
        st.markdown("#### Episode Snapshot")
        st.dataframe(
            episodes_to_dataframe(episodes),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # -----------------------------------------------------
    # Deep-Dive: Episode-by-Episode Intelligence
    # -----------------------------------------------------
    st.subheader("3️⃣ Episode-by-Episode Intelligence")

    for ep in episodes:
        ep_number = ep.get("episode_number", "?")
        ep_title = ep.get("episode_title", "Untitled Episode")
        narrative_breakdown = ep.get("narrative_breakdown", "")

        emotional_arc = ep.get("emotional_arc_analysis", []) or []
        cliffhanger = ep.get("cliffhanger_scoring", {}) or {}
        retention_risks = ep.get("retention_risk_prediction", []) or []
        optimisation_suggestions = ep.get("optimisation_suggestions", []) or []

        # Each episode is presented inside an expander for a compact UI
        with st.expander(f"Episode {ep_number}: {ep_title}", expanded=False):
            # Narrative breakdown gives judges the decomposed story beats
            st.markdown("**Narrative Breakdown**")
            st.write(narrative_breakdown or "_No breakdown provided._")

            # Emotional arc uses time-blocked analysis to show pacing
            st.markdown("**Emotional Arc (per time block)**")
            arc_df = emotional_arc_to_dataframe(emotional_arc)
            if not arc_df.empty:
                st.dataframe(arc_df, use_container_width=True, hide_index=True)
            else:
                st.info("No emotional arc analysis returned for this episode.")

            # Cliffhanger scoring: key for retention between episodes
            st.markdown("**Cliffhanger Strength**")
            c1, c2, c3 = st.columns([1, 1, 3])
            c1.metric("Strength (1–10)", cliffhanger.get("strength_score", "N/A"))
            c2.write(f"**Hook:** {cliffhanger.get('description', 'N/A')}")
            c3.write(f"**Why it works / fails:** {cliffhanger.get('explanation', 'N/A')}")

            # Retention risk prediction: timestamped drop-off forecasts
            st.markdown("**Retention Risk Prediction (drop-off hotspots)**")
            risk_df = retention_risk_to_dataframe(retention_risks)
            if not risk_df.empty:
                st.dataframe(risk_df, use_container_width=True, hide_index=True)
            else:
                st.info("No retention risk predictions returned for this episode.")

            # Optimisation suggestions: actionable guidance for creators
            st.markdown("**Optimisation Suggestions**")
            if optimisation_suggestions:
                for idx, suggestion in enumerate(optimisation_suggestions, start=1):
                    st.markdown(f"- {idx}. {suggestion}")
            else:
                st.info("No optimisation suggestions returned for this episode.")

    # -----------------------------------------------------
    # Raw JSON (for hackathon judges & debugging)
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("4️⃣ Raw JSON Output (for inspection)")
    st.caption(
        "This shows the exact structured payload returned by the Episodic Intelligence Engine, "
        "useful for debugging and verifying schema correctness."
    )
    st.code(json.dumps(parsed, indent=2), language="json")
