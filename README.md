# 🎥 Video Quality Assessment using TimeSformer

## Overview
This project focuses on blind video quality assessment (BVQA), where the goal is to predict how humans perceive video quality without access to a reference video. The output is a Mean Opinion Score (MOS), which reflects perceived quality on a scale of 0–5.

This problem is important for real-world applications such as video streaming platforms, compression systems, and quality monitoring pipelines.

---

## Model Architecture

The model is based on Facebook's TimeSformer, a transformer designed for video understanding.

**Pipeline:**

Input Video (T = 8 frames, 224×224 resolution)  
↓  
Patch Embedding using Conv2D (3 → 768, kernel size 16×16)  
↓  
Spatial & Temporal Positional Embeddings  
↓  
TimeSformer Encoder (12 Transformer layers), each consisting of:
- Spatial Attention (captures distortions within frames)
- Temporal Attention (captures motion artifacts like flickering or frame drops)
- Layer Normalization  
- Feed-Forward Network  
- Residual Connections  
↓  
CLS Token Extraction → shape: (B, 768)  
↓  
Regression Head:
- Linear (768 → 256)  
- ReLU  
- Dropout (0.1)  
- Linear (256 → 1)  
↓  
Predicted MOS Score (0–5)

---

## Dataset

- **Dataset:** KoNViD-1k  
- **Video Resolution:** 1280×720  
- **Frame Rate:** 24/30 FPS  
- **Content:** Diverse real-world scenes with authentic distortions  

- **Size:**
  - Videos: ~2.3 GB  
  - Metadata: ~3 MB  

Dataset Link: https://database.mmsp-kn.de/konvid-1k-database.html

---

## Results

- **PLCC (Pearson Linear Correlation):** 0.6831  
- **SRCC (Spearman Rank Correlation):** 0.6502  
- **RMSE:** 0.4790  
- **MSE:** 0.2295  

These results indicate a reasonable correlation between predicted and human-perceived video quality.

---

## How to Run

### 1. Download Dataset
Download the KoNViD-1k dataset from the official website.

### 2. Configure Paths
Update the paths in the notebook


### 3. Run Notebook
Execute the notebook cells in order:

- Cell 2 — Imports  
- Cell 3 — Configuration  
- Cell 4 — Frame extraction  
- Cell 7 — Dataset & DataLoader creation  
- Cell 8 — Model definition  
- Cell 10 — Training (15 epochs, ~2 hours on T4 GPU)  
- Cells 17–22 — Evaluation and visualization  

---

## Key Learnings

- Understanding transformer-based video models  
- Handling temporal and spatial features in video data  
- Working with real-world noisy datasets  
- Evaluating model performance using correlation metrics  

---

## References

- Bertasius, G., Wang, H., & Torresani, L. (2021). *Is Space-Time Attention All You Need for Video Understanding?* (ICML 2021)  
- Hosu, V. et al. (2017). *The Konstanz Natural Video Database (KoNViD-1k)* (QoMEX 2017)  
- TimeSformer (HuggingFace)  
- Meta AI Blog — TimeSformer  

---

## Author
O. Gayathri Reddy
B.Tech CSE-AIML, Malla Reddy Engineering College

