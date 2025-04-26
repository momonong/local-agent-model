import time, gc, torch, psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# BitsAndBytes 量化設定：4-bit NF4
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
print("🔧 bnb config =", bnb_cfg)

# Tokenizer
t0 = time.time()
tok = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True,
    trust_remote_code=True,   # 建議補上
    local_files_only=True,
)
print(f"🚀 tokenizer loaded (fast={tok.is_fast}) in {time.time()-t0:.2f}s")

# 量化模型
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
    # local_files_only=True,
)
load_sec = time.time() - t0
gpu_name = torch.cuda.get_device_name(model.device)
vram = torch.cuda.memory_allocated(model.device) / 1024**3
print(f"🚀 model loaded to {gpu_name} in {load_sec:.2f}s, "
      f"VRAM now ≈ {vram:.2f} GiB")

# Prompt
prompt = (
    "<|system|>\n你是一個具備推理能力且能分步規劃的 AI 助手。\n"
    "<|user|>\n請列出查詢「台南現在時間」的步驟，不要直接回答時間。\n<|assistant|>"
)
inputs = tok(prompt, return_tensors="pt").to(model.device)
print(f"✏️  prompt tokens = {inputs['input_ids'].shape[-1]}")

# 生成
t0 = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=3,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
gen_sec = time.time() - t0
out_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
print(f"🕒 generate() done in {gen_sec:.2f}s, new tokens = {out_tokens}")

# 解碼 + 顯示
print("\n📤 Model output ↓↓↓\n")
print(tok.decode(outputs[0], skip_special_tokens=True))

# 額外：顯示 Python/系統記憶體
cpu_mem = psutil.Process().memory_info().rss / 1024**3
print(f"\n🖥️  CPU RSS ≈ {cpu_mem:.2f} GiB")
