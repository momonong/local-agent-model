import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 模型設定
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
adapter_path = "model/agent14b-lora"

# 量化載入設定（與訓練時一致）
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Tokenizer 載入
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True,
    trust_remote_code=True,
)

# 模型 + LoRA adapter 載入
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# ✅ 準備 prompt（中英文皆可）
chat = [
    {
        "role": "system",
        "content": "你是一個高效且具備推理能力的 AI 助手，使用繁體中文回答問題。"
                   "你不需要立即給出最終答案，而是需要先進行清晰的推理，"
                   "並分步展示你打算如何解決問題的思考過程。\n\n"
                   "請依照下列格式列出你的思考步驟（不需產出答案，只需規劃）：\n"
                   "<step1>: ...\n<step2>: ...\n<step3>: ...\n\n"
                   "你也可以在步驟中主動使用下列工具來幫助你解決問題：\n"
                   "<tool> search_website(query: str): 從網路搜尋即時資訊並整理摘要。\n"
                   "<tool> get_current_time(): 從 API 取得目前的時間。\n\n"
                   "請記住：你是 agent 模型，你的任務是提出解決問題的完整思考與規劃流程，而不是直接給出結論或執行動作。",
    },
    {"role": "user", "content": "你能幫我查台南現在時間嗎？"}
]

# ✅ chat template 處理成 prompt 字串
prompt = tokenizer.apply_chat_template(chat, tokenize=False)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# ✅ 模型生成
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        num_return_sequences=1,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

# ✅ 解碼與顯示
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("=" * 50)
print("\n📤 模型輸出：\n" + output_text)
