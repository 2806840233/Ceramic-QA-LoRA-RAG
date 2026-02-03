# CeramicQA LoRA + RAG System

本项目致力于构建一个专注于中国陶瓷领域的智能问答系统。项目基于 **Qwen2.5-7B-Instruct** 模型，通过 **LoRA (Low-Rank Adaptation)** 微调技术结合 **RAG (Retrieval-Augmented Generation)** 检索增强生成技术，利用《中国陶瓷史》等专业语料提升模型在陶瓷领域的问答表现。

## 📂 目录结构

```
2026/
├── data/                       # 数据集目录
│   ├── CeramicQA_train.jsonl   # 训练集
│   ├── CeramicQA_val.jsonl     # 验证集
│   ├── CeramicQA_test.jsonl    # 测试集
│   └── 中国陶瓷史_合并_按段落分块.txt # RAG 知识库语料
├── LoRA/                       # LoRA 微调相关
│   ├── train_lora_qwen25_ceramic.py # 微调训练脚本
│   ├── TrainLossPicture.py     # 训练 Loss 可视化
│   ├── ValLossPicture.py       # 验证 Loss 可视化
│   ├── lora_training_loss_final.png # 训练损失曲线
│   ├── lora_eval_loss_final.png     # 验证损失曲线
│   ├── qwen25_ceramic_lora/    # 训练好的 LoRA 权重（忘记添加验证集进行验证loss计算）
│   └── New_qwen25_ceramic_lora/ # 新的 LoRA 权重（包含验证集）
│├── eval_LoRA/                  # LoRA 模型评估（无 RAG）
│   ├── eval_ceramicqa_qwenLoRA.py
│   ├── score.py                # 评分脚本
│   ├── CeramicQA_qwen_lora_preds_val.json # 预测结果
│   └── CeramicQA_qwen_metrics_val.csv      # 评估指标
├── eval_LoRA_RAG/              # LoRA + RAG 模型评估
│   ├── eval_LoRA_RAG.py        # 推理脚本
│   ├── PreSingle.py            # 单样本推理脚本
│   ├── ceramic_faiss.index     # 向量索引文件
│   ├── ceramic_docs.json       # 文档映射文件
│   ├── score.py                # 评分脚本
│   ├── CeramicQA_qwen_lora_rag_preds_val.json # 预测结果
│   └── CeramicQA_qwen_lora_rag_metrics_val.csv # 评估指标
├── eval_QWEN2.5_7B/            # 原始 Qwen 模型评估（基准）
│   ├── eval_ceramicqa_qwen.py  # 推理脚本
│   ├── score.py                # 评分脚本
│   ├── CeramicQA_qwen_preds_val.json # 预测结果
│   └── CeramicQA_qwen_metrics_val.csv      # 评估指标
├── eval_QWEN_RAG/              # 原始 Qwen + RAG 评估
│   ├── eval_RAG.py             # 推理脚本
│   ├── Faiss.py                # 向量索引构建脚本
│   ├── score.py                # 评分脚本
│   ├── ceramic_faiss.index     # 向量索引文件
│   ├── ceramic_docs.json       # 文档映射文件
│   ├── CeramicQA_qwen_rag_preds_val.json # 预测结果
│   └── CeramicQA_qwen_rag_metrics_val.csv # 评估指标
├── CompareWithOtherModel/      # 与其他模型的比较
│   ├── GLM-4-9B-Chat/          # GLM-4 模型评估
│   │   ├── eval_glm4_val.py    # 推理脚本
│   │   ├── score.py            # 评分脚本
│   │   ├── CeramicQA_val_pred_glm.jsonl
│   │   └── CeramicQA_val_metrics_glm.csv
│   └── Yi-1.5-9B-Chat-16K/     # Yi-1.5 模型评估
│       ├── eval_yi15_val.py    # 推理脚本
│       ├── score.py            # 评分脚本
│       ├── CeramicQA_val_pred_yi15.jsonl
│       └── CeramicQA_val_metrics_yi15.csv
├── models/                     # 基础模型存放目录
│   └── Qwen2.5-7B-Instruct/    # Qwen 模型文件
│   └── glm-4-9b-chat/          # GLM-4 模型文件
│   └── Yi-1.5-9b-chat-16k/     # Yi-1.5 模型文件
```

## 🛠️ 环境准备

本项目的运行环境依赖列表已保存在 `requirements.txt` 中。请使用 conda 或 pip 进行安装（建议使用 conda 环境）。

```bash
# 创建并激活 conda 环境
conda create -n qwen_eval python=3.10
conda activate qwen_eval

# 安装基础依赖
conda install --file requirements.txt -c conda-forge

# 补充安装 PyTorch 和其他核心库（根据你的 CUDA 版本调整）
pip install torch transformers peft faiss-cpu sentence-transformers bert-score rouge-score nltk jieba pandas tqdm
```

*注：如果支持 GPU，建议安装对应的 `faiss-gpu` 和 CUDA 版本的 `torch`。*

**基础模型：**
*   LLM: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
*   GLM-4-9B-Chat: [glm-4-9b-chat](https://huggingface.co/THUDM/glme-4-9b-chat)
*   Yi-1.5-9B-Chat-16K: [Yi-1.5-9b-16k](https://huggingface.co/YTHUDM/Yi-1.5-9b-16k-16k)
*   Embedding: [shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese)

## 🚀 使用指南

### 1. 数据准备
将训练数据 (`.jsonl`) 和知识库文本 (`.txt`) 放入 `data/` 目录。
如果知识库文本需要清洗（例如去除过短的段落），可以使用根目录下的工具：
```bash
python filter_blocks.py  # 数据清洗工具脚本，现已删除
```

### 2. 构建 RAG 索引
在进行 RAG 推理之前，需要先对知识库进行向量化并构建 FAISS 索引。（如果知识库有增加新内容或减少内容，需要重新向量化并构建 FAISS 索引）
```bash
cd eval_QWEN_RAG
python Faiss.py
```
生成的 `ceramic_faiss.index` 和 `ceramic_docs.json` 会被用于后续的 RAG 任务。

### 3. LoRA 微调
运行微调脚本开始训练：
```bash
cd LoRA
python train_lora_qwen25_ceramic.py
```
训练完成后，权重将保存在 `LoRA/qwen25_ceramic_lora/` 和 `LoRA/New_qwen25_ceramic_lora/` 目录中。

### 4. 模型评估
本项目提供了多种评估场景，分别对应不同的目录：

*   **原始模型 (Base)**: `cd eval_QWEN2.5_7B && python eval_ceramicqa_qwen.py`
*   **原始模型 + RAG**: `cd eval_QWEN_RAG && python eval_RAG.py`
*   **LoRA 模型**: `cd eval_LoRA && python eval_ceramicqa_qwenLoRA.py`
*   **LoRA + RAG (最终目标)**: `cd eval_LoRA_RAG && python eval_LoRA_RAG.py`
*   **与其他模型比较**: 可在 `CompareWithOtherModel` 目录下评估不同模型的表现
    *   GLM-4-9B-Chat: `cd CompareWithOtherModel/GLM-4-9B-Chat && python eval_glm4_val.py`
    *   Yi-1.5-9B-Chat-16K: `cd CompareWithOtherModel/Yi-1.5-9B-Chat-16K && python eval_yi15_val.py`

### 5. 计算指标
每个评估目录下都有 `score.py`，运行后可计算 BERTScore, ROUGE-1/2/L, METEOR 等指标，并生成 CSV 报告。
```bash
# 例如计算 LoRA + RAG 的分数
cd eval_LoRA_RAG
python score.py
```

### 6. 单样本推理
如果需要进行单样本推理测试，可以使用 `eval_LoRA_RAG` 目录下的 `PreSingle.py` 脚本：
```bash
cd eval_LoRA_RAG
python PreSingle.py
```

## 📊 评估指标说明

*   **BERTScore**: 基于语义相似度的评估。
*   **ROUGE**: 基于 n-gram 重叠的评估 (Recall-Oriented)。
*   **METEOR**: 综合考虑精确率和召回率，支持同义词匹配。

## 📝 备注

*   **微调参数**：r=8, alpha=32, dropout=0.05, lr=2e-4, epochs=3。
*   **RAG 检索**：默认检索 Top-5 相关文档片段作为上下文。
*   **新的 LoRA 权重**：训练好的权重保存在 `LoRA/qwen25_ceramic_lora/` 和 `LoRA/New_qwen25_ceramic_lora/` 目录中。
*   **损失可视化**：使用 `LoRA/TrainLossPicture.py` 和 `LoRA/ValLossPicture.py` 可以生成训练和验证损失曲线。
*   **单样本推理**：`eval_LoRA_RAG/PreSingle.py` 脚本支持单样本问答测试。
*   **与其他模型比较**：`CompareWithOtherModel` 目录包含了 GLM-4-9B-Chat 和 Yi-1.5-9B-Chat-16K 的评估结果。
