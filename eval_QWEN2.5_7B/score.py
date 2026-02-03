import json
import pandas as pd
from bert_score import score as bert_score
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import jieba

# 屏蔽jieba日志（论文代码更干净）
jieba.setLogLevel(jieba.logging.INFO)

# ======================================================
# 1. 配置
# ======================================================
PRED_FILE = "eval_QWEN2.5_7B\CeramicQA_qwen_preds_val.json"
OUT_CSV = "eval_QWEN2.5_7B\CeramicQA_qwen_metrics_val.csv"

# ======================================================
# 2. 读取预测结果
# ======================================================
with open(PRED_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

refs = [x["reference"] for x in data]
preds = [x["prediction"] for x in data]
questions = [x["question"] for x in data]

print(f"读取预测结果: {len(preds)} 条")

# ======================================================
# 3. BERTScore（CPU，稳）
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
# 4. ROUGE（✅ 中文正确版，与单句一致）
# ======================================================
print("计算 ROUGE（中文分词版）...")

rouge = Rouge()

rouge1, rouge2, rougeL = [], [], []

for ref, pred in zip(refs, preds):
    # 中文分词 + 空格拼接
    ref_seg = " ".join(jieba.cut(ref))
    pred_seg = " ".join(jieba.cut(pred))

    scores = rouge.get_scores(
        hyps=pred_seg,
        refs=ref_seg
    )[0]

    rouge1.append(scores["rouge-1"]["f"])
    rouge2.append(scores["rouge-2"]["f"])
    rougeL.append(scores["rouge-l"]["f"])

# ======================================================
# 5. METEOR（中文需分词）
# ======================================================
print("计算 METEOR...")
meteor_scores = []

for ref, pred in zip(refs, preds):
    ref_tok = list(jieba.cut(ref))
    pred_tok = list(jieba.cut(pred))
    meteor_scores.append(
        meteor_score([ref_tok], pred_tok)
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
# 7. 打印论文可用均值
# ======================================================
print("\n=== 平均指标（论文直接用） ===")
for col in [
    "bert_F1",
    "rouge1_F1",
    "rouge2_F1",
    "rougeL_F1",
    "meteor"
]:
    print(f"{col}: {df[col].mean():.4f}")
