
# Clinical Triage Environment (RL)

A reinforcement learning environment that simulates clinical triage decision-making under uncertainty. An agent observes a patient's symptoms, can ask clarifying questions, and must eventually commit to a triage decision — urgency level, department, and recommended next clinical step — while the patient's underlying condition evolves in real time.

The environment is served over HTTP/WebSocket via FastAPI and consumed through a typed Python client, making it usable with any RL training loop or LLM-based agent.

## How It Works

Each episode starts from a case in one of three difficulty datasets (`easy`, `medium`, `hard`). At every step, the agent submits an **action**:

- **`ask`** — request more information; the environment reveals additional (possibly noisy) observations and nudges the "true" disease state forward.
- **`triage`** — commit to a decision: `urgency` (low / medium / high), `department` (e.g. cardiology, neurology, general), and `next_step` (e.g. ECG, CT scan, blood test).

The agent is scored against ground-truth labels attached to each case, with reward shaped by:

- Correctness of urgency, department, and next step
- The patient's simulated **true risk**, which increases over time and is only partially visible to the agent
- Penalties for stalling (asking too long) or triaging too early without enough information
- A "trajectory" bonus for correctly tracking meaningful shifts in patient risk

Difficulty scales the challenge along three axes:

| Task     | Observation noise | Symptom masking | Risk progression | Episode ends when |
|----------|:---:|:---:|:---:|---|
| `easy`   | Low  | Low  | Slow   | Urgency matches ground truth |
| `medium` | Medium | Medium | Moderate | Urgency **and** department match, or 5 steps pass |
| `hard`   | High | High | Fast, with misleading "deceptive" signals | Full match by step 4, or 8 steps pass |

On `hard`, the environment actively injects misleading signals (e.g. "false stability observed," symptom masking that softens "severe" to "moderate") to test whether the agent can reason about hidden deterioration rather than trusting observations at face value.

## Project Structure

```
clinicaltriage/
├── client.py                     # EnvClient implementation (MyEnv) for talking to the server
├── models.py                     # Pydantic schemas: Observation, Action
├── inference.py                  # Example LLM-driven policy loop (rule-based + LLM hybrid)
├── openenv.yaml                  # OpenEnv environment manifest
├── Dockerfile                    # Container build for deploying the environment server
├── requirements.txt / pyproject.toml
├── data/
│   ├── easy.json                 # Ground-truth cases for the easy task
│   ├── medium.json                #   ...medium task
│   └── hard.json                  #   ...hard task
└── server/
    ├── app.py                    # FastAPI app exposing /reset, /step, /health
    ├── my_env_environment.py     # Core environment logic (ClinicalTriageEnv)
    ├── grader.py                 # Standalone scoring utility for evaluation/leaderboards
    └── reward.py                 # Alternate reward function (risk-weighted shaping)
```

## Data Schema

**Observation** — what the agent sees each step:

```python
class Observation(BaseModel):
    case_id: str
    symptoms: str
    history: List[str]
    status: str
```

**Action** — what the agent submits each step:

```python
class Action(BaseModel):
    type: Literal["ask", "triage"]
    question: Optional[str] = None
    urgency: Optional[Literal["low", "medium", "high"]] = None
    department: Optional[Literal["cardiology", "neurology", "pulmonology", "general", "gastroenterology"]] = None
    next_step: Optional[Literal["ECG", "CT scan", "oxygen support", "basic checkup", "ultrasound", "blood test", "rest and hydration"]] = None
    confidence: Optional[float] = None
```

Each case in `data/*.json` provides the ground truth the agent is graded against:

```json
{
  "case_id": "H1",
  "symptoms": "abdominal pain with misleading improvement and recurring nausea",
  "urgency": "medium",
  "department": "gastroenterology",
  "next_step": "ultrasound"
}
```

## Getting Started

### Install

```bash
pip install -r requirements.txt
# or, with uv:
uv sync
```

### Run the environment server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

The server exposes:

| Endpoint | Method | Description |
|---|---|---|
| `/reset?task=easy` | POST | Start a new episode for `easy`, `medium`, or `hard` |
| `/step` | POST | Submit an `Action`, receive the next `Observation`, reward, and done flag |
| `/health` | GET | Health check |

### Connect with the Python client

```python
from client import MyEnv

env = MyEnv(base_url="http://localhost:8000")
result = env.reset()

action = Action(type="ask", question="Any chest pain or shortness of breath?")
result = env.step(action)
```

### Run the example inference loop

`inference.py` runs a simple hybrid policy (keyword-based safety rules + an LLM call for structured triage output) against all three difficulty tiers directly against `ClinicalTriageEnv` (no server required):

```bash
export HF_TOKEN=your_token_here          # or set API_BASE_URL / MODEL_NAME for another provider
python inference.py
```

This prints per-step rewards and a final summary for each task, and serves as a template for wiring up your own policy rule-based, RL-trained, or LLM-based.

### Run with Docker

```bash
docker build -t clinical-triage-env .
docker run -p 8000:8000 clinical-triage-env
```

## Extending the Environment

- **Add cases**: append new entries to `data/easy.json`, `data/medium.json`, or `data/hard.json` following the schema above.
- **Add departments/next steps**: extend the `Literal` options in `models.py`.
- **Tune difficulty**: adjust `noise`, `mask`, and `risk_scale` in `ClinicalTriageEnv.difficulty_config`.
- **Change reward shaping**: modify `_reward` in `server/my_env_environment.py`, or swap in the alternate risk-weighted formulation in `server/reward.py`.

