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
from models import Observation, Action
from server.my_env_environment import ClinicalTriageEnv

app = FastAPI()

env = None
obs = None



@app.post("/reset")
def reset(task: str = "easy"):
    global env, obs

    env = ClinicalTriageEnv(task=task)
    obs = env.reset()

    return {
        "status": "reset done",
        "task": task,
        "case_id": obs.case_id,
        "symptoms": obs.symptoms,
        "history": obs.history
    }


@app.post("/step")
def step(action: Action):
    global env, obs

    if env is None:
        return {"error": "Call /reset first"}

    obs, reward, done, info = env.step(action)

    return {
        "obs": {
            "symptoms": obs.symptoms,
            "history": list(obs.history)
        },
        "reward": reward,
        "done": done,
        "info": info
    }
@app.get("/")
def home():
    return {"status": "Clinical Triage Env Running 🚀"}

def main():
    print("Run with: uvicorn server.app:app --reload")


if __name__ == "__main__":
    main()