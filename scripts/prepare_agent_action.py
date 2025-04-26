# script/prepare_agent_action_en.py
import os, json, re, random
from datasets import load_dataset

SAVE_PATH = "data/agent_action_plan_split.jsonl"
os.makedirs("data", exist_ok=True)
SEED = 42
random.seed(SEED)

def split_prompt(p: str):
    """回傳 (user_text, assistant_text)"""
    # 最後一個 'OUTPUT:' 之後視為 assistant
    if "OUTPUT:" not in p:
        return p.strip(), ""
    user_part, assistant_part = p.rsplit("OUTPUT:", 1)
    # 嘗試抓取方括號 JSON，防止多餘空白
    m = re.search(r"\[[\s\S]+\]", assistant_part)
    assistant_json = m.group(0).strip() if m else assistant_part.strip()
    return user_part.strip(), assistant_json

ds = load_dataset("chats-bug/agent_action_plan", split="train")  # 1 k
rows = []
for i, row in enumerate(ds):
    user_txt, asst_txt = split_prompt(row["prompt"])
    rows.append({
        "id": f"actionplan_en_{i:05d}",
        "messages": [
            {"role": "system", "content": ""},      # 空 system，可自行填
            {"role": "user", "content": user_txt},
            {"role": "assistant", "content": asst_txt},
        ]
    })

random.shuffle(rows)
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ saved → {SAVE_PATH} , total {len(rows)} rows")
