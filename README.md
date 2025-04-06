# 🧠 Lightweight Fine-Tuning with LoRA for Sentiment Classification

This project demonstrates **Parameter-Efficient Fine-Tuning (PEFT)** using **LoRA (Low-Rank Adaptation)** on the `distilbert-base-uncased` model for a binary sentiment classification task using the IMDb movie reviews dataset.

---

## 🚀 Overview

The goal of this project is to showcase how large language models can be efficiently fine-tuned on a specific downstream task without updating all of their parameters. Instead, we only train a small percentage using the **LoRA** adapter via Hugging Face's `peft` library.

---

## 📁 Files Included

| File                  | Description                               |
|-----------------------|-------------------------------------------|
| `LightweightFineTuning.ipynb` | Main notebook with full code, training, and evaluation |
| `peft-imdb.zip`       | Zipped directory containing LoRA adapter weights only (saved via `save_pretrained`) |

---

## 🔍 Model Details

- **Base Model**: [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)
- **PEFT Technique**: [LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- **Library**: Hugging Face Transformers, Datasets, PEFT
- **Task**: Sequence Classification (Sentiment Analysis)
- **Dataset**: IMDb Reviews from Hugging Face Datasets
- **Training Samples**: 2,000 (subset for fast fine-tuning)

---

## 📊 Evaluation Results

| Model        | Accuracy (Eval Set) |
|--------------|---------------------|
| Base Model   | 100% (1.0)          |
| PEFT (LoRA)  | 100% (1.0)          |

📌 Both models achieved perfect accuracy on the 1,000-sample test subset. This shows that LoRA fine-tuning can match full model performance while training fewer than 2% of parameters.

---

## 🛠️ How to Use the Adapter

```python
from peft import AutoPeftModelForSequenceClassification

model = AutoPeftModelForSequenceClassification.from_pretrained("path_to/peft-imdb")
