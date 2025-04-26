import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ 模型 ID（使用 Hugging Face gated model）
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# ✅ Tokenizer 載入
tokenizer = AutoTokenizer.from_pretrained(model_id, token=True, local_files_only=True)
print("Fast tokenizer:", tokenizer.is_fast)

# ✅ 模型載入
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda",
    token=True,
    local_files_only=True,
)

# ✅ 工具描述與 agent-style 指令
system_prompt = f"""
你是一個高效且具備推理能力的 AI 助手，使用繁體中文回答問題。
你不需要立即給出最終答案，而是需要先進行清晰的推理，並分步展示你打算如何解決問題的思考過程。

請依照下列格式列出你的思考步驟（不需產出答案，只需規劃）：
<step1>: ...
<step2>: ...
<step3>: ...

你也可以在步驟中主動使用下列工具來幫助你解決問題：
<tool> search_website(query: str): 從網路搜尋即時資訊並整理摘要。
<tool> get_current_time(): 從 API 取得目前的時間。

請記住：你是 agent 模型，你的任務是提出解決問題的完整思考與規劃流程，而不是直接給出結論或執行動作。

---
User: 你能幫我查台南現在時間嗎？
Assistant:
""".strip()


# ✅ 準備聊天訊息（使用 chat_template）
chat = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "你能幫我查台南現在時間嗎？"},
]

# ✅ Tokenize + 準備輸入
prompt = tokenizer.apply_chat_template(chat, tokenize=False)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# ✅ 模型推論
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        num_return_sequences=1,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

# ✅ 解碼與輸出
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("=" * 50)
print(f"\n📤 模型輸出：\n{output_text}")