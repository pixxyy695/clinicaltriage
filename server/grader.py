def _safe(x: float) -> float:
    """
    Force strict (0,1) bounds required by evaluator.
    """
    return max(1e-6, min(1 - 1e-6, float(x)))


# ---------------- FIELD SCORES ----------------
def check_urgency(pred, truth):
    return 0.99 if pred == truth else 0.01


def check_department(pred, truth):
    return 0.99 if pred == truth else 0.01


def check_next_step(pred, truth):
    return 0.99 if pred == truth else 0.01


# ---------------- FINAL SCORE ----------------
def check_full(action, gt):

    u = check_urgency(action.urgency, gt["urgency"])
    d = check_department(action.department, gt["department"])
    n = check_next_step(action.next_step, gt["next_step"])

    # --- stable aggregation (NO multiplication collapse) ---
    score = (u + d + n) / 3.0

    # optional small bonus shaping (safe, bounded impact)
    if action.type == "ask":
        score += 0.01
    elif action.type == "triage":
        score += 0.02

    # --- final safety clamp (CRITICAL for validator) ---
    return _safe(score)