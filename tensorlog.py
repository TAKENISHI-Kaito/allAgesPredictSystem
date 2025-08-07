from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator("tensorboard/fgnet_logs/version_0/events.out.tfevents.1753544229.AMacBookAir.local.84376.0")
ea.Reload()

print("ログに含まれるタグ:", ea.Tags())

# 例えば accuracy のログを見たい場合
for e in ea.Scalars("val/accuracy"):
    print(f"[Step {e.step}] Accuracy = {e.value:.4f}")