import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======================================================
# 1. 配置
# ======================================================
MODEL_PATH = "/data1/liutao/LiJin/2026/models/Qwen2.5-7B-Instruct"
VAL_FILE = "../data/CeramicQA_val.jsonl"
OUT_JSON = "CeramicQA_qwen_preds_val.json"

BATCH_SIZE = 4
MAX_NEW_TOKENS = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# 2. 加载模型
# ======================================================
print("加载模型中...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()
print("模型加载完成！")

# ======================================================
# 3. 读取验证集
# ======================================================
val_data = []
with open(VAL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        val_data.append(json.loads(line))

total_num = len(val_data)
print(f"验证集大小: {total_num} 条")

# ======================================================
# 4. 读取已完成的预测（断点）
# ======================================================
results = []

if os.path.exists(OUT_JSON):
    with open(OUT_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"检测到已有预测结果: {len(results)} 条，将从断点继续")
else:
    print("未检测到已有结果，从头开始预测")

start_idx = len(results)

# ======================================================
# 5. Batch 推理（断点续跑）
# ======================================================
print(f"从第 {start_idx} 条开始预测...")

for i in tqdm(range(start_idx, total_num, BATCH_SIZE)):
    batch = val_data[i:i + BATCH_SIZE]

    prompts = []
    metas = []

    for item in batch:
        system_prompt = ""
        user_prompt = ""
        gold_answer = ""

        for m in item["messages"]:
            if m["role"] == "system":
                system_prompt = m["content"]
            elif m["role"] == "user":
                user_prompt = m["content"]
            elif m["role"] == "assistant":
                gold_answer = m["content"]

        prompt = f"系统: {system_prompt}\n用户: {user_prompt}\n助手:"
        prompts.append(prompt)

        metas.append({
            "question": user_prompt,
            "reference": gold_answer
        })

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    input_len = inputs["input_ids"].shape[1]

    for j in range(len(prompts)):
        pred_text = tokenizer.decode(
            output_ids[j][input_len:],
            skip_special_tokens=True
        ).strip()

        results.append({
            "question": metas[j]["question"],
            "reference": metas[j]["reference"],
            "prediction": pred_text
        })

    # ===== 🔥 关键：每个 batch 立即落盘 =====
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n全部预测完成，共 {len(results)} 条，结果已保存到 {OUT_JSON}")
