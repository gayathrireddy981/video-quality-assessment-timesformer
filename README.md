# Video Quality Assessment with TimesFormer
Overview 
This project fine-tunes Facebook's TimesFormer for blind video quality assessment(BVAQ) the task of predicting how humans perceive video quality, expressed as a Mean Opinion Score(MOS). 

## Model Architecture 
Input Video (T=8 frames, 224×224)
        ↓
Patch Embedding: Conv2d(3, 768, kernel_size=16×16)
+ Spatial & Temporal Positional Embeddings
        ↓
TimesFormer Encoder (12 Transformer Layers)
  Each layer contains:
  ├─ Spatial Attention (within frames — captures spatial distortions)
  ├─ Temporal Attention (across frames — captures motion flicker, frame drops)
  ├─ Layer Normalization
  ├─ Feed-Forward Network
  └─ Residual Connections
        ↓
Last Hidden State → CLS Token [:, 0, :] → shape: (B, 768)
        ↓
Quality Prediction Head (Regression):
  Linear(768 → 256) → ReLU → Dropout(0.1) → Linear(256 → 1)
        ↓
Predicted MOS Score (0–5 scale)

## Dataset 
📹 Video Specs: 1280×720 resolution, 24/30 fps
🎬 Content Diversity: Various genres & scenes distortions
📂Data Availability: https://database.mmsp-kn.de/konvid-1k-database.html
                               Size : 
                                         Videos: ~2.3 GB
                                         Metadata: ~3 MB 

## Results 
Metric 
PLCC (Pearson Linear Correlation) - 0.6831
SRCC (Spearman Rank Correlation) - 0.6502
RMSE - 0.4790
MSE - 0.2295

## Project Structure 
video-quality-assessment/
│
├── video_quality_assessment.ipynb   # Main notebook (Google Colab)
├── README.md
│
└── Config (inside notebook):
    ├── MODEL_NAME    # facebook/timesformer-base-finetuned-k400
    ├── NUM_FRAMES    # 8
    ├── INPUT_SIZE    # 224
    ├── BATCH_SIZE    # 1
    ├── EPOCHS        # 15
    └── LEARNING_RATE # 1e-4

## How to Run 
1. Get the Dataset
    Download KoNVID-1K from the official site.
2. Configure paths update the Config class in the notebook
   DATA_PATH = "/path/to/konvid1k/videos"
   EXTRACTED_FRAMES_DIR = "/path/to/extracted_frames"
   LABELS_CSV_PATH = "/path/to/konvid1k_metadata.csv"
3. Run the Notebook in Order
   Cell 2 — Imports
   Cell 3 — Config
   Cell 4 — Extract frames from all videos
   Cell 7 — Build Dataset & DataLoaders
   Cell 8 — Define model
   Cell 10 — Train (15 epochs, ~2 hours on T4)
   Cells 17–22 — Evaluate and visualize results

## References
Bertasius, G., Wang, H., & Torresani, L. (2021). Is Space-Time Attention All You Need for Video Understanding? ICML 2021.
Hosu, V. et al. (2017). The Konstanz Natural Video Database (KoNViD-1k). QoMEX 2017.
TimesFormer on HuggingFace
Meta AI Blog — TimesFormer

## Author
O. Gayathri Reddy
B.Tech CSE-AIML, Malla Reddy Engineering College

