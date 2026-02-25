# 🔍 PolitiFact Multimodal Fake News Detection

A deep learning pipeline that detects fake news by combining **text** and **image** signals using state-of-the-art pretrained models.

---

## 📌 Overview

This project builds a multimodal classifier that takes a news article's **text content** and its **associated image** and predicts whether the article is **real** or **fake**. It leverages four powerful models to extract rich, complementary features:

| Model | Role |
|-------|------|
| 🧠 **BERT** (`bert-base-uncased`) | Encodes article text into semantic embeddings |
| 🖼️ **ResNet50** | Extracts visual features from news images |
| 📖 **BLIP** (`blip-image-captioning-base`) | Generates natural language captions from images |
| 🔡 **EasyOCR** | Extracts text embedded within images |

All feature vectors are concatenated and passed through a classification head to produce the final prediction.

---

## 📂 Dataset

The project uses the **PolitiFact AAAI Dataset**, a benchmark dataset for multimodal fake news detection.

- 📊 **Training samples**: 381
- 🧪 **Test samples**: 104
- 🏷️ **Labels**: `0` = Fake News, `1` = Real News

Each sample contains:
- `content` — The text body of the news article
- `image` — Filename of the associated image
- `label` — Ground truth label

> ⚠️ The dataset is **not included** in this repository. You can access it on [Kaggle](https://www.kaggle.com/).

---

## 🏗️ Model Architecture

```
Article Text ──► BERT ──────────────────► [768-dim]  ─┐
News Image   ──► ResNet50 ──────────────► [2048-dim] ─┤
News Image   ──► EasyOCR ──► BERT ──────► [768-dim]  ─┼──► Concat ──► MLP ──► Fake / Real
News Image   ──► BLIP Caption ──► BERT ─► [768-dim]  ─┘
```

Total input feature size: **4352 dimensions**

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/politifact-fake-news-detection.git
cd politifact-fake-news-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure dataset paths
In the notebook, update the following paths to point to your local dataset:
```python
CSV_TRAIN_PATH = "/path/to/politi_train.csv"
CSV_TEST_PATH  = "/path/to/politi_test.csv"
IMG_DIR        = "/path/to/Images"
```

---

## 🚀 Running the Project

Open and run the notebook:
```bash
jupyter notebook politifact-data.ipynb
```

Or run it on **Kaggle** with GPU enabled (recommended — this notebook was originally developed on Kaggle with a GPU accelerator).

---

## 📦 Requirements

```
torch
torchvision
transformers
easyocr
Pillow
pandas
numpy
scikit-learn
matplotlib
tqdm
```

> See `requirements.txt` for exact versions.

---

## 📈 Example Prediction

```python
predict_fake_news(content_text, image_path)

# Output:
# Prediction: FAKE NEWS
# Confidence: 83.75%
# Generated Caption: protesters holding signs outside building
```

---

## 📁 Project Structure

```
📦 politifact-fake-news-detection
 ┣ 📓 politifact-data.ipynb   # Main notebook
 ┣ 📄 README.md               # Project documentation
 ┣ 📄 requirements.txt        # Python dependencies
 ┗ 📁 cache/                  # Cached feature files (auto-generated)
```

---

## 🙏 Acknowledgements

- Dataset: [PolitiFact AAAI Dataset on Kaggle](https://www.kaggle.com/)
- Models: [HuggingFace Transformers](https://huggingface.co/), [PyTorch](https://pytorch.org/), [EasyOCR](https://github.com/JaidedAI/EasyOCR)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
