import os, json, random
import numpy as np
import torch
from datasets import load_dataset

# ---------- 全域隨機種子 ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)      # 如之後要用到 torch DataLoader，可先固定
# ----------------------------------

SAVE_PATH = "data/agent_mix.jsonl"
os.makedirs("data", exist_ok=True)

SOURCES = {             # { HF 路徑 : 要抽的筆數 }
    "YeungNLP/firefly-agent-cot": 20_000,
    "yjernite/function_calls_40k": 6_000,
    "ise-uiuc/CodeAgent-10K":      4_000,
}

# ----------- 各資料集的欄位轉換 -----------
def normalize_firefly(ex):
    user, asst = ex["conversation"][-2:]
    return {"messages": [
        {"role": "user",      "content": user},
        {"role": "assistant", "content": asst.strip()},
    ]}

def normalize_func_call(ex):
    return {"messages": [
        {"role": "user",      "content": ex["question"]},
        {"role": "assistant", "content": ex["answer"]},
    ]}

def normalize_codeagent(ex):
    return {"messages": [
        {"role": "user",      "content": ex["instruction"]},
        {"role": "assistant", "content": ex["output"]},
    ]}

NORMALIZERS = {
    "YeungNLP/firefly-agent-cot": normalize_firefly,
    "yjernite/function_calls_40k": normalize_func_call,
    "ise-uiuc/CodeAgent-10K":      normalize_codeagent,
}
# -------------------------------------------

all_rows = []

for src, n_samples in SOURCES.items():
    print(f"📥  loading {src} …")
    ds = load_dataset(src, split="train", streaming=True)
    ds = ds.shuffle(seed=SEED).take(n_samples)          # 固定 seed 抽樣
    for i, ex in enumerate(ds):
        row = NORMALIZERS[src](ex)
        row["id"] = f"{src.split('/')[-1][:2]}_{i:06d}"
        all_rows.append(row)

print(f"✅ merged rows: {len(all_rows)}")
random.shuffle(all_rows)        # 同樣因為先固定了 random.seed

with open(SAVE_PATH, "w", encoding="utf-8") as f:
    for row in all_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("🚀 saved →", SAVE_PATH)
