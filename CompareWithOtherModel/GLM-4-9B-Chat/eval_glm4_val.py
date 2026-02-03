import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======================================================
# 1. 路径配置
# ======================================================
MODEL_PATH = "/data1/liutao/LiJin/2026/models/glm-4-9b-chat"
VAL_FILE = "../data/CeramicQA_val.jsonl"
OUT_FILE = "./CeramicQA_val_pred_glm.jsonl"

assert torch.cuda.is_available(), "❌ 没检测到 CUDA，请检查 GPU 环境"

# ======================================================
# 2. 加载 tokenizer & model（⚠️ 强制单 GPU）
# ======================================================
print("Loading GLM-4-9B-Chat on single GPU...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

model = model.cuda()
model.eval()

# ======================================================
# 3. 推理函数（稳定版）
# ======================================================
def generate_answer(messages, max_new_tokens=256):
    """
    messages: List[{"role": "system"|"user", "content": str}]
    """

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).cuda()

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in text:
        return text.split("assistant")[-1].strip()
    return text.strip()

# ======================================================
# 4. 读取验证集
# ======================================================
with open(VAL_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Loaded {len(lines)} samples from validation file.")

# ======================================================
# 5. 执行推理
# ======================================================
with open(OUT_FILE, "w", encoding="utf-8") as fout:
    for idx, line in enumerate(tqdm(lines, desc="Inferencing")):
        if idx == 0:
            print(">>> First sample inference starting (should be <10s now)...")

        item = json.loads(line)
        messages = item["messages"]

        infer_messages = [
            m for m in messages if m["role"] in ("system", "user")
        ]

        pred = generate_answer(infer_messages)

        fout.write(json.dumps({
            "messages": messages,
            "prediction": pred
        }, ensure_ascii=False) + "\n")

print(f"\n✅ Done! Saved to {OUT_FILE}")
