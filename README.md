# FinAI-BERT-IslamicBanks

**A Domain-Specific BERT Model for AI Disclosures in Islamic Banking**  
**License**: MIT  
**Model Link**: [Hugging Face - FinAI-BERT-IslamicBanks](https://huggingface.co/bilalzafar/FinAI-BERT-IslamicBanks)  

## Overview
**FinAI-BERT-IslamicBanks** is a fine-tuned BERT-based sentence classification model developed to detect **AI-related disclosures** in the context of **Islamic financial institutions**, specifically Islamic banks. Grounded in the emerging intersection of **AI adoption**, **financial regulation**, and **Islamic finance**, this model supports granular NLP analysis of narrative sections in annual reports.

The model leverages the transformer architecture of `bert-base-uncased`, fine-tuned on manually annotated text from **855 annual reports** issued by **106 Islamic banks** across **25 countries (2015–2024)**. It is intended for use in **research**, **regulatory audits**, **AI readiness benchmarking**, and **ESG-tech discourse analysis**.

## Intended Use
This model is designed to support:

- **Academic research** on AI disclosure patterns in Islamic finance  
- **Regulatory technology (RegTech)** tools for screening annual reports  
- **Technology and ESG audits** of Islamic financial institutions  
- **Index construction** for benchmarking AI-readiness in Islamic banking  
- **Supervised learning baselines** for domain-specific sentence classification tasks  

## 🧠 Model Architecture
- **Base Model**: [`bert-base-uncased`](https://huggingface.co/bert-base-uncased)  
- **Fine-tuned Task**: Binary sentence-level classification (`AI` vs. `Non-AI`)  
- **Tokenizer**: WordPiece tokenizer (standard BERT)  
- **Precision**: Mixed-precision (FP16)  
- **Framework**: Hugging Face Transformers (`Trainer API`)

## Performance Metrics

| Metric         | Score      |
|----------------|------------|
| Accuracy       | 98.67%     |
| F1 Score       | 0.9868     |
| ROC AUC        | 0.9999     |
| Brier Score    | 0.0027     |

The model demonstrates **strong generalization**, **high semantic sensitivity**, and **excellent calibration** across diverse bank report formats.

---

## Training Data
- **Total examples**: 2,632 sentence-level instances  
  - 1,316 AI-related (seed word filtered + manually verified)  
  - 1,316 Non-AI (randomly sampled for balance)  
- **Source documents**: Annual reports from 106 Islamic banks (2015–2024)

Training corpus was extracted using domain-specific lexicons and manually annotated for relevance using a guided coding protocol.

---

## Training Setup

- **Environment**: Google Compute Engine (GPU)
- **Epochs**: 3  
- **Batch size**: 8  
- **Max sequence length**: 128  
- **Optimizer**: AdamW  
- **Loss Function**: Cross-Entropy  
- **Mixed Precision**: Enabled (FP16)  

---

## Repository Contents

| File | Description |
|------|-------------|
| `FinAI-BERT-IslamicBanks - model training.ipynb` | Full model training script |
| `FinAI-BERT-IslamicBanks - training data extraction.ipynb` | Data cleaning & preprocessing pipeline |
| `ai_seedwords.csv` | Seed word list used for initial AI-related sentence extraction |
| `bert_training_data.csv` | Final sentence-level training dataset (manually verified) |


## Model Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bilalzafar/FinAI-BERT-IslamicBanks")
model = AutoModelForSequenceClassification.from_pretrained("bilalzafar/FinAI-BERT-IslamicBanks")

# Classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Label mapping
label_map = {"LABEL_0": "Non-AI", "LABEL_1": "AI"}

# Example
text = "Our Shariah-compliant bank has deployed AI-driven credit risk assessment tools."
result = classifier(text)[0]
label = label_map.get(result['label'], result['label'])
score = result['score']

print(f"Classification: {label} | Score: {score:.4f}")
