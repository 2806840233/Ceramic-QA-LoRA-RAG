# CeramicQA: LoRA + RAG Intelligent Question Answering System for Ceramic Domain  

This project aims to construct an intelligent question answering (QA) system specialized in Chinese ceramic domain. Based on the **Qwen2.5-7B-Instruct** model, the system integrates **LoRA (Low-Rank Adaptation)** fine-tuning and **RAG (Retrieval-Augmented Generation)** technologies, leveraging professional corpora such as *A History of Chinese Ceramics* to enhance the model's QA performance in ceramic-related tasks.  

## 📂 Directory Structure  

```
2026/
├── data/                       # Dataset directory
│   ├── CeramicQA_train.jsonl   # Training set
│   ├── CeramicQA_val.jsonl     # Validation set
│   ├── CeramicQA_test.jsonl    # Test set
│   └── A_History_of_Chinese_Ceramics_Merged_Paragraphs.txt # RAG knowledge base corpus
├── LoRA/                       # LoRA fine-tuning related files
│   ├── train_lora_qwen25_ceramic.py # Fine-tuning training script
│   ├── TrainLossPicture.py     # Training loss visualization script
│   ├── ValLossPicture.py       # Validation loss visualization script
│   ├── lora_training_loss_final.png # Training loss curve
│   ├── lora_eval_loss_final.png     # Validation loss curve
│   ├── qwen25_ceramic_lora/    # Trained LoRA weights (without validation loss calculation)
│   └── New_qwen25_ceramic_lora/ # Updated LoRA weights (with validation set included)
├── eval_LoRA/                  # LoRA model evaluation (without RAG)
│   ├── eval_ceramicqa_qwenLoRA.py # Inference script
│   ├── score.py                # Evaluation metric calculation script
│   ├── CeramicQA_qwen_lora_preds_val.json # Prediction results
│   └── CeramicQA_qwen_metrics_val.csv      # Evaluation metrics report
├── eval_LoRA_RAG/              # LoRA + RAG model evaluation
│   ├── eval_LoRA_RAG.py        # Inference script
│   ├── PreSingle.py            # Single-sample inference script
│   ├── ceramic_faiss.index     # FAISS vector index file
│   ├── ceramic_docs.json       # Document mapping file
│   ├── score.py                # Evaluation metric calculation script
│   ├── CeramicQA_qwen_lora_rag_preds_val.json # Prediction results
│   └── CeramicQA_qwen_lora_rag_metrics_val.csv # Evaluation metrics report
├── eval_QWEN2.5_7B/            # Original Qwen model evaluation (baseline)
│   ├── eval_ceramicqa_qwen.py  # Inference script
│   ├── score.py                # Evaluation metric calculation script
│   ├── CeramicQA_qwen_preds_val.json # Prediction results
│   └── CeramicQA_qwen_metrics_val.csv      # Evaluation metrics report
├── eval_QWEN_RAG/              # Original Qwen + RAG evaluation
│   ├── eval_RAG.py             # Inference script
│   ├── Faiss.py                # FAISS vector index construction script
│   ├── score.py                # Evaluation metric calculation script
│   ├── ceramic_faiss.index     # FAISS vector index file
│   ├── ceramic_docs.json       # Document mapping file
│   ├── CeramicQA_qwen_rag_preds_val.json # Prediction results
│   └── CeramicQA_qwen_rag_metrics_val.csv # Evaluation metrics report
├── CompareWithOtherModel/      # Comparison with other models
│   ├── GLM-4-9B-Chat/          # GLM-4 model evaluation
│   │   ├── eval_glm4_val.py    # Inference script
│   │   ├── score.py            # Evaluation metric calculation script
│   │   ├── CeramicQA_val_pred_glm.jsonl # Prediction results
│   │   └── CeramicQA_val_metrics_glm.csv # Evaluation metrics report
│   └── Yi-1.5-9B-Chat-16K/     # Yi-1.5 model evaluation
│       ├── eval_yi15_val.py    # Inference script
│       ├── score.py            # Evaluation metric calculation script
│       ├── CeramicQA_val_pred_yi15.jsonl # Prediction results
│       └── CeramicQA_val_metrics_yi15.csv # Evaluation metrics report
├── models/                     # Base model storage directory
│   ├── Qwen2.5-7B-Instruct/    # Qwen model files
│   ├── glm-4-9b-chat/          # GLM-4 model files
│   └── Yi-1.5-9b-chat-16k/     # Yi-1.5 model files
```

## 🛠️ Environment Setup  

The project's dependencies are listed in `requirements.txt`. We recommend using a Conda environment for installation:  

```bash
# Create and activate Conda environment
conda create -n qwen_eval python=3.10
conda activate qwen_eval

# Install base dependencies
conda install --file requirements.txt -c conda-forge

# Install PyTorch and core libraries (adjust based on your CUDA version)
pip install torch transformers peft faiss-cpu sentence-transformers bert-score rouge-score nltk jieba pandas tqdm
```

*Note: For GPU support, install `faiss-gpu` and a CUDA-compatible version of `torch`.*  

### Base Models  
- LLM: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)  
- GLM-4-9B-Chat: [glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)  
- Yi-1.5-9B-Chat-16K: [Yi-1.5-9b-chat-16k](https://huggingface.co/01-ai/Yi-1.5-9B-Chat-16K)  
- Embedding Model: [shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese)  

## 🚀 Usage Guide  

### 1. Data Preparation  
Place training data (`.jsonl`) and knowledge base text (`.txt`) into the `data/` directory. For corpus cleaning (e.g., removing overly short paragraphs), use the utility script (removed in the current version):  
```bash
python filter_blocks.py  # Data cleaning script (no longer available)
```

### 2. Build RAG Index  
Before RAG inference, vectorize the knowledge base and construct a FAISS index (re-run if the knowledge base is updated):  
```bash
cd eval_QWEN_RAG
python Faiss.py
```  
Generated files (`ceramic_faiss.index` and `ceramic_docs.json`) will be used for subsequent RAG tasks.  

### 3. LoRA Fine-Tuning  
Run the fine-tuning script to start training:  
```bash
cd LoRA
python train_lora_qwen25_ceramic.py
```  
Trained weights will be saved in `LoRA/qwen25_ceramic_lora/` and `LoRA/New_qwen25_ceramic_lora/`.  

### 4. Model Evaluation  
The project supports multiple evaluation scenarios, corresponding to different directories:  

- **Baseline (Original Model)**:  
  ```bash
  cd eval_QWEN2.5_7B && python eval_ceramicqa_qwen.py
  ```  
- **Original Model + RAG**:  
  ```bash
  cd eval_QWEN_RAG && python eval_RAG.py
  ```  
- **LoRA-Fine-Tuned Model**:  
  ```bash
  cd eval_LoRA && python eval_ceramicqa_qwenLoRA.py
  ```  
- **LoRA + RAG (Target System)**:  
  ```bash
  cd eval_LoRA_RAG && python eval_LoRA_RAG.py
  ```  
- **Comparison with Other Models**:  
  Evaluate competing models in the `CompareWithOtherModel` directory:  
  - GLM-4-9B-Chat:  
    ```bash
    cd CompareWithOtherModel/GLM-4-9B-Chat && python eval_glm4_val.py
    ```  
  - Yi-1.5-9B-Chat-16K:  
    ```bash
    cd CompareWithOtherModel/Yi-1.5-9B-Chat-16K && python eval_yi15_val.py
    ```  

### 5. Calculate Evaluation Metrics  
Each evaluation directory contains a `score.py` script to compute metrics (BERTScore, ROUGE-1/2/L, METEOR) and generate CSV reports:  
```bash
# Example: Calculate metrics for LoRA + RAG
cd eval_LoRA_RAG
python score.py
```  

### 6. Single-Sample Inference  
For single-query testing, use the `PreSingle.py` script in `eval_LoRA_RAG`:  
```bash
cd eval_LoRA_RAG
python PreSingle.py
```  

## 📊 Evaluation Metrics Explanation  

- **BERTScore**: Semantic similarity-based evaluation metric.  
- **ROUGE**: Recall-oriented metric based on n-gram overlap.  
- **METEOR**: Balances precision and recall with synonym matching support.  

## 📝 Notes  

- **Fine-Tuning Hyperparameters**: r=8, alpha=32, dropout=0.05, learning rate=2e-4, epochs=3.  
- **RAG Retrieval**: Default to top-5 most relevant document chunks as context.  
- **LoRA Weights**: Trained weights are stored in `LoRA/qwen25_ceramic_lora/` (without validation) and `LoRA/New_qwen25_ceramic_lora/` (with validation).  
- **Loss Visualization**: Use `LoRA/TrainLossPicture.py` and `LoRA/ValLossPicture.py` to generate training/validation loss curves.  
- **Single-Sample Inference**: `eval_LoRA_RAG/PreSingle.py` supports ad-hoc QA testing.  
- **Model Comparison**: The `CompareWithOtherModel` directory includes evaluation results for GLM-4-9B-Chat and Yi-1.5-9B-Chat-16K.  

---