import wandb
import numpy as np

api = wandb.Api()
run = api.run("samithuang/flow_grpo/2u7ed2w0")
system_metrics = run.history(stream="events", pandas=True)

df = system_metrics[["_timestamp", "system.proc.memory.rssMB"]].dropna().sort_values("_timestamp")
cpu_rss = df["system.proc.memory.rssMB"].values

# subsample to 10 points
indices = np.linspace(0, len(cpu_rss)-1, 10, dtype=int)
print("CPU Memory (RSS MB) over time (10 samples throughout training):")
for i, idx in enumerate(indices):
    print(f"Sample {i+1} ({idx}/{len(cpu_rss)}): {cpu_rss[idx]:.1f} MB")

