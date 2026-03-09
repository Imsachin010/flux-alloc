from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

log_dir = "ppo_allocator_logs/PPO_1"

ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

tags = ea.Tags()["scalars"]

data = []

for tag in tags:
    events = ea.Scalars(tag)
    for e in events:
        data.append({
            "metric": tag,
            "step": e.step,
            "value": e.value
        })

df = pd.DataFrame(data)

df.to_csv("tensorboard_metrics.csv", index=False)

print("Saved tensorboard_metrics.csv")