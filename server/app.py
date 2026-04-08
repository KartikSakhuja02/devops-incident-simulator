"""
server/app.py — FastAPI server for the DevOps Incident Response Simulator.

Exposes the environment via HTTP endpoints so the OpenEnv client
and the hackathon validators can interact with it.

Endpoints:
    POST /reset        — Start a new episode
    POST /step         — Take an action
    GET  /state        — Get current episode metadata
    GET  /health       — Health check (validators ping this)
    GET  /tasks        — List all available tasks with descriptions
"""

import sys
import os

# Make sure models.py (one level up) is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Package import works for uvicorn server.app:app in Docker; fallback keeps direct script runs working.
try:
    from .devops_incident_simulator_environment import DevOpsEnvironment
except ImportError:
    from devops_incident_simulator_environment import DevOpsEnvironment
from models import IncidentAction, LogObservation, StepResult, EpisodeState


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DevOps Incident Response Simulator",
    description=(
        "An OpenEnv-compatible environment where an AI agent reads pipeline logs "
        "and takes remediation actions to fix incidents."
    ),
    version="1.0.0",
)

# Allow all origins so the HF Space and validators can reach the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One shared environment instance per server process
# (fine for hackathon / single-agent use)
env = DevOpsEnvironment()


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None   # "task_1" | "task_2" | "task_3" | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Health check endpoint.
    The hackathon validator pings this — must return 200.
    """
    return {"status": "ok", "environment": "devops-incident-simulator"}


@app.get("/tasks")
def list_tasks():
    """
    Returns the list of available tasks with difficulty and description.
    Useful for the inference script and for judges reviewing the environment.
    """
    return {
        "tasks": [
            {
                "task_id": "task_1",
                "difficulty": "easy",
                "description": (
                    "A single service is throwing an obvious error (e.g. "
                    "NullPointerException or OOMKill). Identify the service "
                    "and apply the correct fix."
                ),
            },
            {
                "task_id": "task_2",
                "difficulty": "medium",
                "description": (
                    "A deployment is stuck or a config update broke a service. "
                    "The pipeline is partially deployed. You must roll back "
                    "or redeploy to restore service."
                ),
            },
            {
                "task_id": "task_3",
                "difficulty": "hard",
                "description": (
                    "A cascading failure has taken down multiple services. "
                    "Identify the ROOT CAUSE service (not the symptoms) "
                    "and apply the correct remediation."
                ),
            },
        ]
    }


@app.post("/reset", response_model=LogObservation)
def reset(request: ResetRequest = ResetRequest()):
    """
    Start a new episode. Optionally specify task_id to force a difficulty.
    Returns the initial observation (logs + pipeline status).
    """
    try:
        observation = env.reset(task_id=request.task_id)
        return observation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: IncidentAction):
    """
    Submit an action. Returns a StepResult with:
        - new observation (updated logs)
        - reward (0.0 to 1.0)
        - done flag
        - feedback message
    """
    try:
        result = env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=EpisodeState)
def state():
    """
    Returns current episode metadata:
        episode_id, step_count, task_difficulty, cumulative_reward, etc.
    """
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Run directly (for local testing without Docker)
# ---------------------------------------------------------------------------

def main():
    """Entrypoint for running the API server as a script."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()