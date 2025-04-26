from transformers import TrainerCallback

# 自訂一個 Callback，保存最佳模型
class SaveBestModelCallback(TrainerCallback):
    def __init__(self):
        self.best_loss = float("inf")

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" in logs:
            current_loss = logs["loss"]
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                print(f"🌟 New best loss: {self.best_loss:.4f}, saving checkpoint...")
                control.should_save = True   # 強制存一版