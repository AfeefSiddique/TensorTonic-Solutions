def schedule_pipeline(tasks, resource_budget):
    # Build lookup structures
    task_map  = {t["name"]: t for t in tasks}
    deps      = {t["name"]: set(t["depends_on"]) for t in tasks}
    completed = set()
    scheduled = {}   # name -> start_time
    running   = []   # list of (end_time, name)

    time = 0

    while len(scheduled) < len(tasks):
        # 1. Complete tasks whose end_time <= current time
        just_done = [name for end, name in running if end <= time]
        for name in just_done:
            completed.add(name)
        running = [(end, name) for end, name in running if end > time]

        # 2. Identify ready tasks
        used = sum(task_map[name]["resources"] for _, name in running)
        ready = sorted(
            [t["name"] for t in tasks
             if t["name"] not in scheduled
             and deps[t["name"]].issubset(completed)],
        )

        # 3. Greedily assign alphabetically
        for name in ready:
            cost = task_map[name]["resources"]
            if used + cost <= resource_budget:
                scheduled[name] = time
                running.append((time + task_map[name]["duration"], name))
                used += cost

        # 4. Advance time to next completion event
        if running:
            time = min(end for end, _ in running)

    return sorted(scheduled.items(), key=lambda x: (x[1], x[0]))