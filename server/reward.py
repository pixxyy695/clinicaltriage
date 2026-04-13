def _reward(self, action, gt):

    r = 0.01

    urgency_map = {"low": 0.3, "medium": 0.6, "high": 0.99}

    # correctness
    if action.urgency == gt["urgency"]:
        r += 0.4

    if action.department == gt["department"]:
        r += 0.3

    if action.next_step == gt["next_step"]:
        r += 0.2

    
    risk = self.state["risk_score"]

    if action.urgency == "low" and risk > 0.6:
        r -= 0.7  

    if action.urgency == "high" and risk < 0.3:
        r -= 0.2  
    # delayed outcome shaping
    r += risk * 0.2

    # step penalty
    r -= 0.03 * self.step_count

    return max(0.01, min(0.99, r))