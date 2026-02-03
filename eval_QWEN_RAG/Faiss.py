import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# =====================
# 配置
# =====================
DOC_FILE = "../data/中国陶瓷史_合并_按段落分块.txt"
FAISS_INDEX = "ceramic_faiss.index"
DOC_STORE = "ceramic_docs.json"

EMBED_MODEL = "shibing624/text2vec-base-chinese"

# =====================
# 1. 读取分块文本
# =====================
docs = []
with open(DOC_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if len(line) > 20:
            docs.append(line)

print(f"Loaded {len(docs)} paragraphs")

# =====================
# 2. 向量化
# =====================
model = SentenceTransformer(EMBED_MODEL)
embeddings = model.encode(
    docs,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype("float32")

# =====================
# 3. 构建 FAISS
# =====================
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # 余弦相似度
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX)

# =====================
# 4. 保存原文
# =====================
with open(DOC_STORE, "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print("FAISS index built successfully.")
