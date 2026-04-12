def check_urgency(pred, truth):
    return 0.99 if pred == truth else 0.01


def check_department(pred, truth):
    return 0.99 if pred == truth else 0.01


def check_next_step(pred, truth):
    return 0.99 if pred == truth else 0.01


def check_full(action, gt):

    return (
        check_urgency(action.urgency, gt["urgency"]) *
        check_department(action.department, gt["department"]) *
        check_next_step(action.next_step, gt["next_step"])
    )