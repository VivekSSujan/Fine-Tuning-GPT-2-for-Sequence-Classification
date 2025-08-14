# GPT‑2 Fine‑Tuning for Tweet Sentiment (Neutral / Positive / Sad)

This repository contains a Jupyter Notebook that fine‑tunes **GPT‑2** to classify **tweets** into three sentiments:
- `neutral`
- `positive`
- `sad`

Built with the Hugging Face `transformers` ecosystem, the notebook walks through data loading, preprocessing, training, evaluation, and inference.

---

## 🔎 What’s Inside
- **Data preprocessing**: cleaning tweets, tokenization
- **Label mapping**: `neutral=0`, `positive=1`, `sad=2` (configurable)
- **Model setup**: GPT‑2 adapted for sequence classification (`AutoModelForSequenceClassification`)
- **Training**: `Trainer` API with early stopping & weighted loss (optional)
- **Evaluation**: accuracy, precision/recall/F1 (macro & weighted), confusion matrix
- **Inference**: predict sentiment for new tweets

---
### Prerequisites
- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Datasets](https://huggingface.co/docs/datasets/index)
- GPU support recommended for faster training
