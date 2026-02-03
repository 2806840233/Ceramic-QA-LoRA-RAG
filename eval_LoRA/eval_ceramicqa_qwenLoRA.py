import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ======================================================
# 1. 配置
# ======================================================
BASE_MODEL_PATH = "/data1/liutao/LiJin/2026/models/Qwen2.5-7B-Instruct"
LORA_PATH = "../LoRA/qwen25_ceramic_lora"
VAL_FILE = "../data/CeramicQA_val.jsonl"
OUT_JSON = "CeramicQA_qwen_lora_preds_val.json"

BATCH_SIZE = 4
MAX_NEW_TOKENS = 256

# ======================================================
# 2. tokenizer（关键）
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True
)

tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ======================================================
# 3. 模型 + LoRA
# ======================================================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

model.eval()
print("✅ Model + LoRA loaded")

# ======================================================
# 4. 读取验证集
# ======================================================
data = []
with open(VAL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

print(f"📄 Validation samples: {len(data)}")

# ======================================================
# 5. Batch 推理（正确 Chat Template）
# ======================================================
results = []

for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Evaluating"):
    batch = data[i:i + BATCH_SIZE]

    prompts = []
    metas = []

    for sample in batch:
        # ✅ 直接用 messages
        chat_prompt = tokenizer.apply_chat_template(
            sample["messages"][:-1],   # ❗去掉 assistant 的参考答案
            tokenize=False,
            add_generation_prompt=True
        )

        prompts.append(chat_prompt)

        question = ""
        reference = ""
        for m in sample["messages"]:
            if m["role"] == "user":
                question = m["content"]
            elif m["role"] == "assistant":
                reference = m["content"]

        metas.append({
            "question": question,
            "reference": reference
        })

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    for j in range(len(prompts)):
        output_text = tokenizer.decode(
            outputs[j][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        results.append({
            "question": metas[j]["question"],
            "reference": metas[j]["reference"],
            "prediction": output_text
        })

# ======================================================
# 6. 保存
# ======================================================
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n🎉 Done! Saved to {OUT_JSON}")
