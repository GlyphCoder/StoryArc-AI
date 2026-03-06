# VBOX Episodic Intelligence Engine

AI-powered narrative intelligence for creators designing **multi-part vertical series**.

This engine turns a single short story idea into a fully-analysed 5–8 episode 90‑second vertical series, providing:

- **Story Decomposer Engine** – episode-wise arc from a single idea.
- **Emotional Arc Analyser** – time-blocked emotional shifts and flat engagement zones.
- **Cliffhanger Strength Scoring** – 1–10 scoring with rationale.
- **Retention Risk Predictor** – predicted drop-off hotspots within 90 seconds.
- **Optimisation Suggestion Engine** – structured, episode-level improvements.

---

## High-level architecture

Vertical separation between **intelligence layer (Python + Gemini)** and **creator workspace UI (HTML/CSS/JS)**.

```mermaid
flowchart LR
    C[Creator Browser UI\nHTML + CSS + JS] -->|story idea, params| API[/FastAPI Backend/]
    API -->|calls| LLM[Google Gemini\nGenerativeModel]
    LLM -->|JSON (strict schema)| Engine[Python Episodic Engine\n(post-processing, scoring)]
    Engine -->|series + metrics| API
    API -->|JSON| C
```

---

## Project structure

- `backend/episodic_engine.py` – core Episodic Intelligence Engine (LLM call + JSON parsing + heuristic metrics).
- `backend/api.py` – FastAPI server exposing `/api/analyse`.
- `frontend/index.html` – main creator-facing UI.
- `frontend/styles.css` – modern, glassmorphism-inspired styling.
- `frontend/app.js` – JS client calling the backend and rendering analytics.
- `app.py` – optional Streamlit prototype UI using the same idea (can be demoed separately).
- `.env` – environment config (Gemini key placeholder).
- `requirements.txt` – Python dependencies.

---

## Quick start (end‑to‑end)

### 1. Set your Gemini API key

Edit `.env` at the project root:

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

