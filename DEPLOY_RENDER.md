# Public deployment on Render

The project is deployment-ready as a **single Docker web service**. The container runs FastAPI internally on port 8000 and exposes only the redesigned Streamlit UI on Render's public `$PORT`.

## 1. Push this clean folder to GitHub
Do not add `.env` or `venv/`. They are already ignored.

## 2. Render
1. Render Dashboard -> **New +** -> **Blueprint** (or Web Service).
2. Connect the GitHub repository containing this folder.
3. If using Blueprint, Render reads `render.yaml` automatically.
4. Add secret environment variable `GEMINI_API_KEY` with your Gemini key.
5. Deploy.

`STARTUP_SEED=true` resets the synthetic demo dataset whenever the service starts, keeping the hackathon scenario deterministic.

## 3. Share the Render URL
Judges only see the Streamlit product UI. FastAPI stays internal to the container.

## Optional Neo4j
The app works without Neo4j. For a hosted Neo4j Aura instance, add `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` as Render secrets.
