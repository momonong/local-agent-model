# script/train_lora.py

import os, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from utils.save_best_model import SaveBestModelCallback

# ====== 設定區 ======
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DATA_PATH = "data/agent_action_plan_split.jsonl"
MAX_LEN = 1024
BATCH = 1
GRAD_ACC = 4
LR = 5e-5
OUT_DIR = "models/deepseek14b-lora"
LOG_DIR = "logs/deepseek14b-lora"
EPOCHS = 40
LOGGING_STEPS = 10
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 2
# =====================

# ✅ 4-bit 量化設定
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ✅ 載入 tokenizer（會當作 processing_class 傳入）
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
    trust_remote_code=True,
)

# ✅ 載入 base 模型 + LoRA
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

# ✅ 載入與處理資料集
ds = load_dataset("json", data_files=DATA_PATH, split="train")

def to_text(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

ds = ds.map(to_text, remove_columns=ds.column_names)

# ✅ 訓練參數設定（新版用 SFTConfig）
sft_config = SFTConfig(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    max_seq_length=MAX_LEN,
    fp16=True,
    packing=True,
    logging_dir=LOG_DIR,
    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    report_to=["tensorboard"],
)

# ✅ 初始化 Trainer（新版參數）
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds,
    processing_class=tokenizer,
    callback=[SaveBestModelCallback(),
              EarlyStoppingCallback(early_stopping_patience=10)],
)

# ✅ 開始訓練與保存 LoRA adapter
trainer.train(resume_from_checkpoint=True)
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
