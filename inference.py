import os
import json
from openai import OpenAI
from server.my_env_environment import ClinicalTriageEnv
from models import Action
from dotenv import load_dotenv

load_dotenv()

# ---------------- ENV VARS ----------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN")


client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


# ---------------- LLM POLICY ----------------
def get_action(obs):

    # 🟢 ASK PHASE (ONLY EARLY)
    if len(obs.history) < 2:
        return {
            "type": "ask",
            "question": "Any symptoms or vitals?"
        }

    # 🧠 SYMPTOM ANALYSIS (ALWAYS AVAILABLE)
    sym = obs.symptoms.lower()

    # 🚨 SAFETY RULES (HIGHEST PRIORITY)
    if any(k in sym for k in ["chest", "heart"]):
        urgency = "high"
    elif any(k in sym for k in ["pain", "fever", "severe"]):
        urgency = "medium"
    else:
        urgency = "low"

    # 🧠 LLM PROMPT
    prompt = f"""
Return JSON only:
- urgency: low, medium, high
- department: cardiology, neurology, pulmonology, gastroenterology, general
- next_step: ECG, CT scan, oxygen support, ultrasound, basic checkup

Symptoms:
{obs.symptoms}
History:
{obs.history}
"""

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        data = json.loads(res.choices[0].message.content)
    except:
        data = {}

    # 🧠 FINAL SAFE OUTPUT
    return {
        "type": "triage",
        "urgency": data.get("urgency", urgency),
        "department": data.get("department", "general"),
        "next_step": data.get("next_step", "basic checkup")
    }
# ---------------- MAIN LOOP ----------------
def run():
    for task in ["easy", "medium", "hard"]:
        env = ClinicalTriageEnv(task)
        obs = env.reset()

        print(f"[START] task={task} env=clinical_triage model={MODEL_NAME}")

        rewards = []
        step = 0
        done = False

        while not done and step < 6:
            step += 1

            action_dict = get_action(obs)

            action = Action(
                type=action_dict.get("type", "triage"),
                urgency=action_dict.get("urgency", "medium"),
                department=action_dict.get("department", "general"),
                next_step=action_dict.get("next_step", "basic checkup"),
                question=action_dict.get("question", "")
            )

            obs, reward, done, info = env.step(action)

            rewards.append(f"{reward:.2f}")

            error = info.get("error")
            if error is None:
                error = "null"

            print(
                f"[STEP] step={step} action={action.type} "
                f"reward={reward:.2f} done={str(done).lower()} error={error}"
            )

        print(
            f"[END] success={str(done).lower()} steps={step} "
            f"rewards={','.join(rewards)}"
        )


if __name__ == "__main__":
    run()