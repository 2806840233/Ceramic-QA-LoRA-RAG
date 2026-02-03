import json
import pandas as pd
from bert_score import score as bert_score
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import jieba

jieba.setLogLevel(jieba.logging.INFO)

# ======================================================
# 1. 配置
# ======================================================
PRED_FILE = r"CompareWithOtherModel\Yi-1.5-9B-Chat-16K\CeramicQA_val_pred_yi15.jsonl"
OUT_CSV = r"CompareWithOtherModel\Yi-1.5-9B-Chat-16K\CeramicQA_val_metrics_yi15.csv"

# ======================================================
# 2. 读取 JSONL（⚠️ 关键修正点）
# ======================================================
data = []
with open(PRED_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

questions = [x["messages"][1]["content"] for x in data]
refs = [x["messages"][2]["content"] for x in data]
preds = [x["prediction"] for x in data]

print(f"读取预测结果: {len(preds)} 条")

# ======================================================
# 3. BERTScore
# ======================================================
print("计算 BERTScore（CPU）...")
P, R, F1 = bert_score(
    preds,
    refs,
    lang="zh",
    device="cpu",
    verbose=True
)

# ======================================================
# 4. ROUGE（中文分词，论文标准）
# ======================================================
print("计算 ROUGE（中文分词版）...")
rouge = Rouge()

rouge1, rouge2, rougeL = [], [], []

for ref, pred in zip(refs, preds):
    ref_seg = " ".join(jieba.cut(ref))
    pred_seg = " ".join(jieba.cut(pred))

    scores = rouge.get_scores(pred_seg, ref_seg)[0]

    rouge1.append(scores["rouge-1"]["f"])
    rouge2.append(scores["rouge-2"]["f"])
    rougeL.append(scores["rouge-l"]["f"])

# ======================================================
# 5. METEOR
# ======================================================
print("计算 METEOR...")
meteor_scores = []

for ref, pred in zip(refs, preds):
    meteor_scores.append(
        meteor_score(
            [list(jieba.cut(ref))],
            list(jieba.cut(pred))
        )
    )

# ======================================================
# 6. 保存 CSV
# ======================================================
df = pd.DataFrame({
    "question": questions,
    "reference": refs,
    "prediction": preds,
    "bert_F1": F1.tolist(),
    "rouge1_F1": rouge1,
    "rouge2_F1": rouge2,
    "rougeL_F1": rougeL,
    "meteor": meteor_scores
})

df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

# ======================================================
# 7. 论文可直接使用的均值
# ======================================================
print("\n=== 平均指标（论文直接用） ===")
for col in ["bert_F1", "rouge1_F1", "rouge2_F1", "rougeL_F1", "meteor"]:
    print(f"{col}: {df[col].mean():.4f}")
