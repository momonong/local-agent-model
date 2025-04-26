import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ✅ 模型 ID
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
# adapter_path = "models/llama4-scout-lora"  # 如果你有微調 LoRA adapter 的話

# ✅ 量化設定
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ✅ 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_fast=True,
)

# ✅ 載入模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_cfg,
)

# model = PeftModel.from_pretrained(base_model, adapter_path)
# model.eval()

# 載入系統指令
system_prompt = """
你是一個具備推理能力且善於拆解問題的 AI 助手。
請不要直接給出最終答案，而是分步驟（<step1>、<step2>、<step3>...）展示你的思考與行動計畫。
每個步驟可以包含需要查找或推理的行動，保持條理清晰。
請只根據目前使用者的問題推理，不要延續、參考或受之前問題影響。
你不需要生成最終回答，最終回答將由其他模型完成。
請確保輸出使用繁體中文，且內容只包含步驟，格式保持一致且簡潔。

以下是範例：

使用者問題：
「世界各國最近在氣候變遷議題上有什麼新政策？」

助手示範回答：
<step1>: 搜尋聯合國氣候變遷組織（UNFCCC）和各國官方網站，收集最近一個月的政策更新。
<step2>: 依據不同國家分類，例如美國、歐盟、中國，整理各自的氣候行動內容。
<step3>: 分析各國新政策對全球碳排放減量目標的可能影響與趨勢。
"""

# 使用者問題
user_prompt = "川普最近在關稅上發表了什麼看法？"

# Chat 組裝
chat = [
    {"role": "system", "content": system_prompt.strip()},
    {"role": "user", "content": user_prompt.strip() + "請列出思考步驟。"},
]

# ✅ Tokenize
inputs = tokenizer.apply_chat_template(chat, tokenize=False)
inputs = tokenizer(inputs, return_tensors="pt").to(model.device)

# ✅ 推理
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

# ✅ 解碼
result = tokenizer.decode(output[0], skip_special_tokens=True)
print("=" * 50)
print(result)
