import json
import os
import torch
import faiss
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# ======================================================
# 1. 配置
# ======================================================
MODEL_PATH = "/data1/liutao/LiJin/2026/models/Qwen2.5-7B-Instruct"
VAL_FILE = "../data/CeramicQA_val.jsonl"
OUT_JSON = "CeramicQA_qwen_rag_preds_val.json"

FAISS_INDEX = "ceramic_faiss.index"
DOC_STORE = "ceramic_docs.json"
EMBED_MODEL = "shibing624/text2vec-base-chinese"

TOP_K = 5
BATCH_SIZE = 1
MAX_NEW_TOKENS = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# 2. 加载模型
# ======================================================
print("加载 Qwen 模型...")
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
print("Qwen 模型加载完成")

# ======================================================
# 3. 加载 Faiss & 向量模型
# ======================================================
print("加载 Faiss 索引...")
index = faiss.read_index(FAISS_INDEX)

with open(DOC_STORE, "r", encoding="utf-8") as f:
    doc_store = json.load(f)

embed_model = SentenceTransformer(EMBED_MODEL)

def retrieve_context(question, top_k=5):
    q_emb = embed_model.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    _, indices = index.search(q_emb, top_k)

    return [doc_store[i] for i in indices[0] if i != -1]

# ======================================================
# 4. 读取验证集
# ======================================================
val_data = []
with open(VAL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        val_data.append(json.loads(line))

# 只取前 12 条用于快速验证
# val_data = val_data[:12]
# print(f"当前用于预测的样本数: {len(val_data)}")

# ======================================================
# 5. 断点续跑
# ======================================================
results = []
if os.path.exists(OUT_JSON):
    with open(OUT_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)

start_idx = len(results)
print(f"从第 {start_idx} 条开始预测")

# ======================================================
# 6. RAG 推理
# ======================================================
for i in tqdm(range(start_idx, len(val_data), BATCH_SIZE)):
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

        contexts = retrieve_context(user_prompt, TOP_K)
        context_text = "\n".join([f"{idx+1}. {c}" for idx, c in enumerate(contexts)])

        prompt = (
            f"系统: {system_prompt}\n"
            f"已知资料:\n{context_text}\n\n"
            f"用户问题: {user_prompt}\n"
            f"请直接给出问题的最终答案，只输出答案本身。"
            f"不要复述问题，不要解释推理过程，"
            f"不要出现“系统”“用户”“资料”“参考”“来源”等任何说明性文字，"
            f"不要添加背景、评价或总结性语句。\n"
            f"助手:"
        )

        prompts.append(prompt)
        metas.append({
            "question": user_prompt,
            "reference": gold_answer,
            "contexts": contexts
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

    # ==================================================
    # ★ FIX：使用 attention_mask 计算每条样本真实 prompt 长度
    # ==================================================
    for j in range(len(prompts)):
        prompt_len = inputs["attention_mask"][j].sum().item()

        pred = tokenizer.decode(
            output_ids[j][prompt_len:],
            skip_special_tokens=True
        )

        # ==================================================
        # ★ FIX：只取第一条非空有效答案行（评测级清洗）
        # ==================================================
        pred = next(
            (line.strip() for line in pred.splitlines() if line.strip()),
            ""
        )

        results.append({
            "question": metas[j]["question"],
            "reference": metas[j]["reference"],
            "prediction": pred,
            "contexts": metas[j]["contexts"]
        })

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

print("RAG 推理完成")
