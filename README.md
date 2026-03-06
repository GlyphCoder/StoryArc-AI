# VBOX – Episodic Intelligence Engine

> 🎬 AI-powered platform for creating and analyzing short-form vertical series with advanced narrative intelligence.

![VBOX](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.12+-blue) ![React](https://img.shields.io/badge/React-18-61dafb) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-009688)

## 🚀 Overview

VBOX is an AI-powered episodic intelligence engine that transforms your story ideas into fully analyzed, production-ready vertical series (90-second episodes). It leverages **Google Gemini AI** to provide comprehensive narrative analysis and creative guidance.

## 🎬 Features

### Core Analysis Engine
- **Story Decomposer** – Episode-wise narrative arc from a single idea
- **Emotional Arc Analyzer** – Time-blocked emotional shifts with flat engagement zone detection
- **Cliffhanger Strength Scoring** – 1–10 scoring with detailed rationale
- **Retention Risk Predictor** – Predicted drop-off hotspots within 90-second episodes
- **Optimization Suggestions** – Structured, episode-level improvement recommendations

### Enhanced Creative Tools (NEW!)
- **Character Development Engine** – Detailed character arcs, backgrounds, visual descriptions
- **Dialogue Suggestion System** – Natural, punchy dialogue optimized for short-form format
- **Visual Mood Board Generator** – Color palettes, camera styles, lighting design, location suggestions
- **Music & Sound Design Recommender** – Genre, tempo, reference artists, sound effects
- **Shot Composition Guide** – Frame techniques, camera movements, equipment suggestions
- **Multi-Format Genre Support** – Drama, Comedy, Thriller, Romance, Sci-Fi, Fantasy

### Modern Interface
- **Beautiful React Frontend** – Glassmorphism design with dark theme
- **Interactive Visualizations** – Charts for emotional arcs and risk prediction
- **Tab-Based Navigation** – Organized info across Overview, Episodes, Characters, Visuals, Music, Shots
- **Responsive Design** – Works on desktop and tablet
- **Real-time API Integration** – Seamless communication with backend

### Text-to-Speech (Optional)
- **Google Cloud TTS Integration** – Professional voice narration preview
- **Multiple Voice Options** – Various languages and speaker profiles
- **Base64 Audio Stream** – Direct audio playback in browser

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (React Frontend)                  │
│           Modern UI with Chart.js Visualizations             │
└────────────────────────────┬────────────────────────────────┘
                             │
                    HTTP/JSON API
                             │
┌────────────────────────────▼────────────────────────────────┐
│                   FastAPI Server                              │
│  /api/analyse  /api/characters  /api/dialogue  /api/tts      │
│  /api/mood-board  /api/music  /api/shots  /api/health       │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
        ▼                     ▼                      ▼
    [Gemini AI]         [Episodic Engine]    [Google Cloud TTS]
  (Text Generation)     (JSON Parsing &      (Voice Synthesis)
                        Metrics)
```

---

## 📂 Project Structure

```
.
├── backend/
│   ├── api.py                  # FastAPI endpoints for all features
│   ├── episodic_engine.py      # Core & extended creative engines
│   └── __pycache__/
├── frontend/
│   ├── index.html              # Main React application (NEW!)
│   ├── index_old.html          # Legacy version (archived)
│   ├── styles.css              # Custom styles
│   └── [other components]
├── app.py                      # Optional Streamlit prototype
├── list_gemini_models.py       # Utility script
├── requirements.txt            # Python dependencies
├── .env                        # Configuration (API keys)
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Google Gemini API

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY="your_actual_gemini_api_key_here"
```

Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 3. (Optional) Configure Text-to-Speech

For Google Cloud Text-to-Speech:

```bash
# Install gcloud CLI and authenticate
gcloud auth application-default login

# (The backend will use your authenticated credentials)
```

### 4. Start the Backend Server

```bash
cd backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 5. Open the Frontend

Open your browser to your frontend hosting (e.g., simple HTTP server):

```bash
cd frontend
python -m http.server 8001
# Visit http://localhost:8001
```

Or open `frontend/index.html` directly in your browser (API calls will work cross-origin).

---

## 📝 API Endpoints

### Main Analysis
**POST** `/api/analyse`
- Generates complete series analysis with all features
- Request body:
  ```json
  {
    "core_idea": "A detective hunting a mysterious figure in the city",
    "episode_count": 6,
    "genre": "thriller",
    "tone": "suspenseful",
    "target_audience": "18-35",
    "model_name": "gemini-2.5-flash"
  }
  ```

### Extended Features
- **POST** `/api/characters` – Character development only
- **POST** `/api/dialogue` – Dialogue suggestions
- **POST** `/api/mood-board` – Visual mood board
- **POST** `/api/music` – Music recommendations
- **POST** `/api/shots` – Shot composition

### Text-to-Speech
**POST** `/api/tts`
```json
{
  "text": "Your dialogue or narration text",
  "voice_name": "en-US-Neural2-C",
  "language_code": "en-US"
}
```

### Health Check
**GET** `/api/health` – Returns `{"status": "healthy", "version": "2.0.0"}`

---

## 🎨 Features Showcase

### Emotional Arc Analysis
- Visual breakdown of emotions across time blocks
- Engagement level tracking (High/Medium/Low)
- Automatic flat zone detection with warnings
- Interactive doughnut charts showing emotion distribution

### Character Development
- Full character profiles with archetypes
- Character evolution across episodes
- Visual appearance suggestions
- Ensemble dynamics analysis
- Expandable character cards with trait badges

### Visual Mood Board
- Hex color palettes with reasoning
- Camera and lighting recommendations
- Location suggestions
- VFX and transition ideas
- Typography and text overlay guidance

### Music & Sound Design
- Opening theme specifications (genre, tempo, mood)
- Background music by emotional state
- Reference artists and genres
- Sound effects library
- Silence/breathing room placement

### Shot Composition
- Time-blocked shot breakdowns
- Camera movement specifications
- Depth of field recommendations
- Focus pulling techniques
- Equipment suggestions

---

## 💻 Technology Stack

**Backend:**
- FastAPI (async web framework)
- Google Generative AI SDK (Gemini)
- Google Cloud Text-to-Speech
- Pydantic (validation)
- Uvicorn (ASGI server)

**Frontend:**
- React 18
- Tailwind CSS (utility-first CSS)
- Chart.js (interactive visualizations)
- Lucide Icons
- Framer Motion (animations)

**Deployment-Ready:**
- CORS enabled for cross-origin requests
- Environment-based configuration
- Async/await patterns for performance

---

## 🎯 Use Cases

1. **Content Creators** – Plan vertical series for social media
2. **Screenwriters** – Develop episodic story structures
3. **Producers** – Guide production with AI-generated insights
4. **Educators** – Teach storytelling with data-driven feedback
5. **Narrative Designers** – Design interactive story branching

---

## 🔧 Advanced Configuration

### Model Selection
Change the Gemini model by updating `model_name` in the request:
```json
{
  "model_name": "gemini-1.5-pro"
}
```

Available models:
- `gemini-2.5-flash` (default, fast)
- `gemini-1.5-pro` (more detailed)
- `gemini-1.5-flash` (faster)

### Custom Voices (TTS)
Google Cloud TTS supports many voices:
- `en-US-Neural2-A` through `en-US-Neural2-I` (various genders/ages)
- Multilingual support with language codes (e.g., `fr-FR`, `es-ES`, `ja-JP`)

### Environment Variables
```bash
GEMINI_API_KEY=your_key_here
# Optional: Google Cloud credentials for TTS
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

---

## 📊 Performance Tips

- **Episode Count:** Start with 6 episodes; more episodes = longer processing
- **Model Selection:** `gemini-2.5-flash` is fastest for hackathons
- **Batch Requests:** The backend can handle concurrent requests
- **Caching:** Consider caching results for the same story idea

---

## ⚠️ Limitations & Future Work

- **Episode Limit:** Currently 5–8 episodes per series
- **Video Previews:** Shot composition doesn't generate actual footage (future integration with Runway/Synthesia)
- **TTS Voices:** Requires Google Cloud authentication for now
- **Real-time Streaming:** API responses are full objects (could be streamed for large responses)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is provided as-is for educational and commercial use.

---

## 🎓 Credits

- **Google Gemini API** – AI narrative intelligence
- **Google Cloud TTS** – Voice synthesis
- **FastAPI** – Backend framework
- **React + Tailwind** – Frontend architecture

---

## 🚀 Getting Help

- Check the API documentation at `/docs` (Swagger UI)
- Review sample outputs in the results tabs
- Test with the provided example prompts

**Happy storytelling! 🎬**


```bash
GEMINI_API_KEY="YOUR_REAL_GEMINI_KEY_HERE"
```

> Your key must have access to a text generation model such as `gemini-2.5-flash`.

### 2. Install Python dependencies

From the project root:

```bash
cd HACKNUTHON
pip install -r requirements.txt
```

### 3. Run the backend API

Start the Episodic Intelligence API on port 8000:

```bash
uvicorn backend.api:app --reload --port 8000
```

You can verify it with:

```bash
curl http://127.0.0.1:8000/health
```

You should see:

```json
{"status": "ok"}
```

### 4. Launch the web UI

Serve the static frontend and open it in your browser:

```bash
cd frontend
python -m http.server 5173
```

Then open:

```text
http://127.0.0.1:5173
```

The web UI is configured to talk to the backend at `http://127.0.0.1:8000/api/analyse`.

---

## How each required module is implemented

### 1. Story Decomposer Engine

- **Where**: `backend/episodic_engine.py → generate_episodic_intelligence`.
- **How**:
  - Uses a strong **system prompt** (`EPISODIC_SYSTEM_PROMPT`) instructing Gemini to:
    - Break a single idea into a **5–8 episode** vertical series (`total_episodes`, `episodes[]`).
    - Assume **90 seconds per episode**.
  - User prompt (`_build_user_prompt`) pins **exact episode count** requested by the creator.
  - Model returns a strict JSON object with:
    - `series_title`, `series_logline`, `total_episodes`, and per-episode `narrative_breakdown`.

### 2. Emotional Arc Analyser

- **Where**:
  - LLM output: `episodes[].emotional_arc_analysis[]`.
  - Post-processing: `_compute_engine_metrics` (flatness stats).
- **How**:
  - For each episode, Gemini outputs **time-blocked emotional states**:
    - `time_block` (e.g. `"0–15s"`, `"15–45s"`).
    - `dominant_emotion`, `engagement_level`, `flat_zone_warning`.
  - The engine then:
    - Counts **flat-zone warnings** across all episodes.
    - Computes a **flatness index per episode** = (# flat blocks / # total blocks).
    - Flags the **flattest episode** (where pacing is most at risk).

### 3. Cliffhanger Strength Scoring

- **Where**: `episodes[].cliffhanger_scoring`.
- **How**:
  - Gemini is instructed via JSON schema to return:
    - `description` – textual summary of the cliffhanger hook.
    - `strength_score` – integer 1–10.
    - `explanation` – why the cliffhanger will or won’t retain viewers.
  - Engine composes an **average cliffhanger strength** metric over all episodes.
  - Frontend highlights:
    - Per-episode `Cliffhanger score: X/10`.
    - Series-level **average cliffhanger score** in the metrics bar.

### 4. Retention Risk Predictor

- **Where**: `episodes[].retention_risk_prediction[]`.
- **How**:
  - Gemini returns per-episode retention predictions:
    - `drop_off_timestamp` within the 90-second window.
    - `risk_level` string (Low/Medium/High).
    - `reason_for_dropoff` natural language reason.
  - Engine heuristics:
    - Counts **high-risk segments** and **medium-risk segments**.
    - Surfaces these counts as series-wide metrics.
  - Frontend:
    - Renders **risk pills** per episode:
      - Red for High.
      - Yellow for Medium.
      - Green for Low.
    - Tooltip shows `reason_for_dropoff`, giving judges clear interpretability.

### 5. Optimisation Suggestion Engine

- **Where**: `episodes[].optimisation_suggestions[]`.
- **How**:
  - Gemini is asked to output **actionable suggestions** for improving:
    - Emotional arc pacing.
    - Cliffhanger punch.
    - Retention at predicted drop-off zones.
  - Frontend turns this into a concise **per-episode checklist** so creators know:
    - What to cut.
    - What to amplify.
    - Where to move beats for better retention.

---

## Why this is more than a “GPT wrapper”

1. **Hard JSON contract**  
   - Uses `response_mime_type="application/json"` and a strict schema.
   - Python backend **validates and post-processes** the JSON, instead of just proxying raw text.

2. **Heuristic narrative metrics** (`_compute_engine_metrics`)
   - Aggregates:
     - Average cliffhanger strength.
     - High/medium risk segments.
     - Total flat-zone warnings.
     - Flattest episode based on flatness index.
   - These metrics are derived from the structure of the JSON, not hallucinated text.

3. **Clear separation of concerns**
   - Intelligence layer (`backend/episodic_engine.py`) can be swapped with:
     - Custom ML models.
     - Hybrid rules + embeddings.
   - UI and delivery are completely decoupled (FastAPI + static frontend).

4. **Explainability**
   - Every risk and score has:
     - A timestamp.
     - A textual reason.
     - An aggregated metric visible at the series level.

---

## How to use it as a creator

1. Open the web UI.
2. Paste a short‑form series idea into the story input.
3. Choose how many episodes you want (5–8).
4. Run the engine.
5. Explore:
   - **Series overview**: title, logline, global metrics.
   - **Per‑episode insights**:
     - Emotional arc over the 90‑second window.
     - Cliffhanger strength and explanation.
     - Predicted drop‑off hotspots.
     - Concrete optimisation suggestions.
   - **Raw JSON**: machine‑readable structure for integrating into your own tools.

---

## Optional: Streamlit prototype

- You can still run the original Streamlit prototype:

  ```bash
  streamlit run app.py
  ```

- This provides a second UI to demonstrate the same intelligence in a more “dashboard-like” environment directly in Python.

