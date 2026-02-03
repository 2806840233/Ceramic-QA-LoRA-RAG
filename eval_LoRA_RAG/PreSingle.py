import json
import torch
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# ======================================================
# 1. 配置
# ======================================================
BASE_MODEL_PATH = "/data1/liutao/LiJin/2026/models/Qwen2.5-7B-Instruct"
LORA_PATH = "../LoRA/qwen25_ceramic_lora"

FAISS_INDEX = "ceramic_faiss.index"
DOC_STORE = "ceramic_docs.json"
EMBED_MODEL = "shibing624/text2vec-base-chinese"

TOP_K = 3
MAX_NEW_TOKENS = 256

# ======================================================
# 2. 加载 tokenizer & 模型
# ======================================================
print("加载模型...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

# ======================================================
# 3. 加载 FAISS & 向量模型
# ======================================================
index = faiss.read_index(FAISS_INDEX)

with open(DOC_STORE, "r", encoding="utf-8") as f:
    doc_store = json.load(f)

embed_model = SentenceTransformer(EMBED_MODEL)

def retrieve_context(question, top_k=3):
    q_emb = embed_model.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")
    _, indices = index.search(q_emb, top_k)
    return [doc_store[i] for i in indices[0] if i != -1]

# ======================================================
# 4. 单条测试问题（你只改这里）
# ======================================================
system_prompt = "你是一个严谨的中国陶瓷知识问答助手，只基于给定资料作答。"

question = "陶瓷烧制过程中，窑温的控制对成品有什么影响？"

# ======================================================
# 5. 构建 RAG 内容
# ======================================================
contexts = retrieve_context(question, TOP_K)

context_text = "\n".join(
    [f"{i+1}. {c}" for i, c in enumerate(contexts)]
)

user_content = (
    f"已知资料：\n{context_text}\n\n"
    f"问题：{question}\n"
    f"请基于已知资料作答。"
)

# ======================================================
# 6. 使用 Qwen 官方 chat template（核心）
# ======================================================
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_content}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True   # ★ 关键：强制进入 assistant 生成态
).to(model.device)

print("\n================ PROMPT（反序列化后） ================\n")
print(tokenizer.decode(input_ids[0], skip_special_tokens=False))
print("\n=====================================================\n")

# ======================================================
# 7. 推理
# ======================================================
with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# ======================================================
# 8. 解码（无需手算 prompt_len）
# ======================================================
generated_ids = output_ids[0][input_ids.shape[1]:]

pred = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True
).strip()

print("=============== 模型回答 ===============\n")
print(pred if pred else "【空输出】")
print("\n=======================================\n")
