import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ======================================================
# 1. 路径配置
# ======================================================
MODEL_PATH = "/data1/liutao/LiJin/2026/models/Qwen2.5-7B-Instruct"
TRAIN_FILE = "../data/CeramicQA_train.jsonl"
EVAL_FILE  = "../data/CeramicQA_test.jsonl"
OUTPUT_DIR = "./New_qwen25_ceramic_lora"

# ======================================================
# 2. LoRA 配置
# ======================================================
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ======================================================
# 3. tokenizer & model
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",          # ✅ 保留 auto
    trust_remote_code=True
)

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.print_trainable_parameters()

# ======================================================
# 4. 数据集
# ======================================================
train_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
eval_dataset  = load_dataset("json", data_files=EVAL_FILE,  split="train")

# ======================================================
# 5. tokenize
# ======================================================
def tokenize_fn(example):
    text = ""
    for msg in example["messages"]:
        if "role" in msg:
            text += f"{msg['role']}: "
        text += msg["content"] + " "
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=2048,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = train_dataset.map(tokenize_fn, batched=False)
eval_dataset  = eval_dataset.map(tokenize_fn, batched=False)

# ======================================================
# 6. TrainingArguments（transformers 4.57.5）
# ======================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,       # ✅ eval batch
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,

    logging_steps=10,
    save_steps=10,
    save_total_limit=2,

    eval_strategy="steps",
    eval_steps=10,

    report_to="none",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03
)

# ======================================================
# 7. SFTTrainer
# ======================================================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# ======================================================
# 8. 开始训练（⚠️ 不断点续训）
# ======================================================
# checkpoint_path = "/data1/liutao/LiJin/2026/LoRA/New_qwen25_ceramic_lora/checkpoint-100/"

# trainer.train(resume_from_checkpoint=checkpoint_path)

trainer.train()

# ======================================================
# 9. 保存 LoRA
# ======================================================
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter 已保存到 {OUTPUT_DIR}")
