# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the My Env Environment.

This module creates an HTTP server that exposes the MyEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from models import Action, Observation
from server.my_env_environment import ClinicalTriageEnv
from threading import Lock

app = FastAPI()

env = None
obs = None
lock = Lock()


def _safe(x: float) -> float:
    return max(0.0001, min(0.9900, float(x)))


# ---------------- RESET ----------------
@app.post("/reset")
def reset(task: str = "easy"):
    global env, obs

    env = ClinicalTriageEnv(task=task)
    obs = env.reset()

    return {
        "status": "reset done",
        "task": task,
        "observation": {
            "case_id": obs.case_id,
            "symptoms": obs.symptoms,
            "history": obs.history,
            "status": obs.status
        }
    }


# ---------------- STEP ----------------
@app.post("/step")
def step(action: Action):
    global env, obs

    if env is None:
        return {"error": "Call /reset first"}

    with lock:
        obs, reward, done, info = env.step(action)

    return {
        "observation": {
            "case_id": obs.case_id,
            "symptoms": obs.symptoms,
            "history": list(obs.history),
            "status": obs.status
        },
        "reward": _safe(reward),
        "done": done,
        "info": info
    }


# ---------------- HEALTH ----------------
@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------------- REDIRECT ROOT → DOCS ----------------
@app.get("/")
def home():
    return RedirectResponse(url="/docs")


# ---------------- FIX HF /web → DOCS ----------------
@app.get("/web")
def web():
    return RedirectResponse(url="/docs")


def main():
    print("Run with: uvicorn server.app:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()