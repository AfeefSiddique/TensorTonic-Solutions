def retraining_policy(daily_stats, config):
    drift_threshold      = config["drift_threshold"]
    performance_threshold = config["performance_threshold"]
    max_staleness        = config["max_staleness"]
    cooldown             = config["cooldown"]
    retrain_cost         = config["retrain_cost"]
    budget               = config["budget"]

    days_since_retrain = 0
    last_retrain_day   = -cooldown   # cooldown initially satisfied
    retrain_days       = []

    for stat in daily_stats:
        day         = stat["day"]
        drift       = stat["drift_score"]
        performance = stat["performance"]

        days_since_retrain += 1

        # Check trigger conditions
        triggered = (
            drift > drift_threshold or
            performance < performance_threshold or
            days_since_retrain >= max_staleness
        )

        # Check constraints
        cooldown_ok = (day - last_retrain_day) >= cooldown
        budget_ok   = budget >= retrain_cost

        if triggered and cooldown_ok and budget_ok:
            retrain_days.append(day)
            budget            -= retrain_cost
            last_retrain_day   = day
            days_since_retrain = 0

    return retrain_days