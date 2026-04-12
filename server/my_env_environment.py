import json
import random
from models import Observation, Action


class ClinicalTriageEnv:

    def __init__(self, task="easy"):
        self.task_name = task

        with open(f"data/{task}.json") as f:
            raw = json.load(f)

        self.data = list(raw.values()) if isinstance(raw, dict) else raw

        self.index = 0
        self.state = None
        self.step_count = 0

        self.difficulty_config = {
            "easy":   {"noise": 0.05, "mask": 0.05, "risk_scale": 1.0},
            "medium": {"noise": 0.15, "mask": 0.25, "risk_scale": 1.3},
            "hard":   {"noise": 0.35, "mask": 0.60, "risk_scale": 1.8},
        }

    # ---------------- RESET ----------------
    def reset(self):
        item = self.data[self.index % len(self.data)]
        self.index += 1
        self.step_count = 0

        self.state = {
            "case_id": item["case_id"],
            "base_symptoms": item["symptoms"],
            "true_severity": item["urgency"],
            "observed_symptoms": item["symptoms"],
            "true_risk": 0.2,
            "history": [],
            "ground_truth": item,
            "last_action": None,
            "progression_step": 0,
            "trajectory_score": 0.0,
            "last_risk_bucket": 0,
        }

        return self._obs()

    # ---------------- STEP ----------------
    def step(self, action: Action):

        self.step_count += 1
        gt = self.state["ground_truth"]
        cfg = self.difficulty_config[self.task_name]

        reward = 0.0
        done = False

        try:
            self._progress_disease()

            if action.type == "ask":

                self.state["history"].append(action.question or "")

                info_gain = random.uniform(0.05, 0.2)
                if self.task_name == "hard":
                    info_gain += 0.18

                reward = info_gain - cfg["noise"]

                hints = [
                    "uncertain progression",
                    "possible hidden deterioration",
                    "conflicting vitals",
                    "monitor closely"
                ]
                self.state["observed_symptoms"] += " | " + random.choice(hints)

            else:
                reward = self._reward(action, gt)
                done = self._done(action, gt)

            self.state["last_action"] = action.type

            return self._obs(), max(0.0, min(1.0, reward)), done, {"error": None}

        except Exception as e:
            return self._obs(), 0.0, True, {"error": str(e)}

    # ---------------- DISEASE MODEL (LEADERBOARD CORE) ----------------
    def _progress_disease(self):

        cfg = self.difficulty_config[self.task_name]

        severity_map = {"low": 0.08, "medium": 0.25, "high": 0.55}
        base = severity_map.get(self.state["true_severity"], 0.2)

        noise = random.uniform(-0.25, 0.35) * cfg["noise"]

        self.state["true_risk"] += base * cfg["risk_scale"] + noise

        # HARD MODE escalation
        if self.task_name == "hard":

            self.state["true_risk"] += 0.04 * (self.state["progression_step"] ** 1.25)

            if random.random() < 0.35:
                self.state["true_risk"] += random.uniform(0.06, 0.2)

            deception = [
                "false stability observed",
                "conflicting clinical signals",
                "temporary improvement (unreliable)",
                "hidden deterioration likely"
            ]

            if random.random() < 0.6:
                self.state["observed_symptoms"] += " | " + random.choice(deception)

        self.state["true_risk"] = max(0.0, min(1.0, self.state["true_risk"]))

        # trajectory signal (IMPORTANT)
        bucket = 0 if self.state["true_risk"] < 0.3 else 1 if self.state["true_risk"] < 0.6 else 2
        if bucket != self.state["last_risk_bucket"]:
            self.state["trajectory_score"] += 0.05
            self.state["last_risk_bucket"] = bucket

        self.state["progression_step"] += 1

    # ---------------- OBS ----------------
    def _obs(self):

        cfg = self.difficulty_config[self.task_name]
        symptoms = self.state["observed_symptoms"]

        if random.random() < cfg["mask"]:
            symptoms = symptoms.replace("severe", "moderate")
            symptoms = symptoms.replace("acute", "uncertain")

        return Observation(
            case_id=self.state["case_id"],
            symptoms=symptoms,
            history=self.state["history"],
            status="open"
        )

    # ---------------- DONE ----------------
    def _done(self, action, gt):

        if self.task_name == "easy":
            return (
                action.type == "triage"
                and action.urgency == gt["urgency"]
            )

        if self.task_name == "medium":
            return (
                action.urgency == gt["urgency"]
                and action.department == gt["department"]
            ) or self.step_count >= 5

        correct = (
            action.urgency == gt["urgency"]
            and action.department == gt["department"]
            and action.next_step == gt["next_step"]
        )

        if self.task_name == "hard":
            return (correct and self.step_count >= 4) or self.step_count >= 8

        return correct

    # ---------------- REWARD ----------------
    def _reward(self, action, gt):

        r = 0.0
        risk = self.state["true_risk"]

        # correctness
        if action.urgency == gt["urgency"]:
            r += 0.3
        if action.department == gt["department"]:
            r += 0.2
        if action.next_step == gt["next_step"]:
            r += 0.2

        # safety
        if gt["urgency"] == "high" and action.urgency == "low":
            r -= 0.4

        # risk calibration
        if risk > 0.7:
            r += 0.4 if action.urgency == "high" else -0.3

        elif risk < 0.3:
            r += 0.1 if action.urgency == "low" else -0.2

        # trajectory bonus
        r += 0.2 * self.state["trajectory_score"]

        # HARD MODE logic
        if self.task_name == "hard":

            if action.type == "ask":
                r += 0.2

        # reduce harsh early penalties
            if self.step_count <= 2 and action.type == "triage":
                r -= 0.2   

            if len(self.state["history"]) == 0 and action.type == "triage":
                r -= 0.3   

        # reward correct late decisions
            if self.step_count >= 3 and action.type == "triage":
                r += 0.2

        # small step penalty (not destructive)
            r -= 0.02 * self.step_count

        return float(max(-1.0, min(1.0, r)))
        def state(self):
            return self.state