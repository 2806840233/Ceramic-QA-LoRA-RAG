import json
import os
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# ======================================================
# 1. 配置
# ======================================================
BASE_MODEL_PATH = "/data1/liutao/LiJin/2026/models/Qwen2.5-7B-Instruct"
LORA_PATH = "../LoRA/qwen25_ceramic_lora"
VAL_FILE = "../data/CeramicQA_val.jsonl"
OUT_JSON = "CeramicQA_qwen_lora_rag_preds_val.json"

FAISS_INDEX = "ceramic_faiss.index"
DOC_STORE = "ceramic_docs.json"
EMBED_MODEL = "shibing624/text2vec-base-chinese"

TOP_K = 3
BATCH_SIZE = 1  # RAG情况下保持batch_size=1
MAX_NEW_TOKENS = 256

# ======================================================
# 2. 加载tokenizer和LoRA模型
# ======================================================
print("加载tokenizer和LoRA模型...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True
)

# 设置padding
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载基础模型 + LoRA
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
print("LoRA模型加载完成")

# ======================================================
# 3. 加载Faiss索引和向量模型
# ======================================================
print("加载Faiss索引...")
index = faiss.read_index(FAISS_INDEX)

with open(DOC_STORE, "r", encoding="utf-8") as f:
    doc_store = json.load(f)

embed_model = SentenceTransformer(EMBED_MODEL)
print("向量模型加载完成")

def retrieve_context(question, top_k=5):
    """检索相关上下文"""
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

# 可以控制实验规模
# val_data = val_data[:12]  # 只取前12条进行快速实验
print(f"验证集样本数: {len(val_data)}")

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
# 6. LoRA + RAG 推理
# ======================================================
for i in tqdm(range(start_idx, len(val_data), BATCH_SIZE), desc="推理进度"):
    batch = val_data[i:i + BATCH_SIZE]

    prompts = []
    metas = []

    for item in batch:
        # 提取系统提示、用户问题和参考答案
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

        # 检索相关上下文
        contexts = retrieve_context(user_prompt, TOP_K)
        context_text = "\n".join([f"{idx+1}. {c}" for idx, c in enumerate(contexts)])

        # 构建RAG提示词
        prompt = (
            f"系统: {system_prompt}\n"
            f"已知资料:\n{context_text}\n\n"
            f"用户问题: {user_prompt}\n"
            f"请直接给出问题的最终答案，只输出答案本身，不能不回复。"
            f"助手请回答:"
        )

        prompts.append(prompt)
        metas.append({
            "question": user_prompt,
            "reference": gold_answer,
            "contexts": contexts
        })

    # Tokenize - 特别关注padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    # 将input_ids转移到模型所在设备
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # 生成答案
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0
        )

    # 解码并保存结果
    for j in range(len(prompts)):
        # ★ 关键修复：使用attention_mask计算实际prompt长度
        # 原代码问题：inputs["input_ids"].shape[1] 返回的是batch中的最大长度
        # 而不是每个样本的实际长度
        prompt_len = attention_mask[j].sum().item()
        
        # 提取生成的tokens（排除prompt部分）
        generated_ids = output_ids[j][prompt_len:]
        
        # 解码
        pred = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        # 打印调试信息（可选）
        print(f"DEBUG - 样本 {i+j}:")
        print(f"  Prompt长度: {prompt_len}")
        print(f"  原始解码: '{pred}'")
        
        # 进一步清理：移除可能的多余换行和空白
        pred_lines = [line.strip() for line in pred.splitlines() if line.strip()]
        if pred_lines:
            pred = pred_lines[0]  # 取第一行
        else:
            pred = ""
            
        # 额外的清理：移除可能的助手标签
        unwanted_prefixes = ["助手:", "assistant:", "回答:", "答案:"]
        for prefix in unwanted_prefixes:
            if pred.startswith(prefix):
                pred = pred[len(prefix):].strip()
        
        # print(f"  清理后: '{pred}'")

        results.append({
            "question": metas[j]["question"],
            "reference": metas[j]["reference"],
            "prediction": pred,
            "contexts": metas[j]["contexts"],
            "model": "qwen25-lora-rag"
        })

    # 每处理完一个batch就保存
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

print("\n✅ LoRA+RAG 推理完成！")
print(f"结果已保存到: {OUT_JSON}")
print(f"预测样本数: {len(results)}")