# 🔍 VisionInspect

## Unsupervised Industrial Defect Detection & Localization

VisionInspect is a computer vision system for **industrial anomaly detection and defect localization** using normal-only learning, deep CNN feature embeddings, and PatchCore-style nearest-neighbour matching.

The system learns the visual distribution of normal industrial products and identifies abnormal regions without requiring labeled examples of every possible defect.

It currently supports three categories from the **MVTec Anomaly Detection dataset**:

- 🍾 Bottle
- 🌰 Hazelnut
- 🧱 Tile

VisionInspect supports both **image inspection** and **video inspection** through an interactive Streamlit application.

---

## 🚀 Live Project

### Live AI Demo

https://visioninspect-aidetection.streamlit.app

### Project Showcase

https://muhammadtalha-pk.github.io/VisionInspect/

### Source Code

https://github.com/MuhammadTalha-pk/VisionInspect

---

## ✨ Main Features

### 🖼️ Image Inspection

Upload an image and receive:

- NORMAL / ANOMALOUS classification
- Image-level anomaly score
- Category-specific calibrated threshold
- Patch-level anomaly heatmap
- Defect localization overlay

### 🎯 Defect Localization

VisionInspect reconstructs local PatchCore anomaly scores into a spatial heatmap.

The interface displays:

1. Input image
2. Anomaly heatmap
3. Localization overlay

### 🎬 Video Inspection

VisionInspect also supports sampled-frame video analysis.

The application can:

- Sample every Nth video frame
- Inspect each sampled frame independently
- Count anomalous frames
- Calculate anomaly rate
- Track anomaly scores over time
- Display a temporal anomaly profile
- Identify the most anomalous frame
- Generate heatmap localization for the peak frame

### 📊 Research Dashboard

The application includes:

- Three-category benchmark comparison
- Autoencoder vs PatchCore comparison
- Coreset ablation study
- Inference-time trade-off
- Raw experimental results
- Architecture explanation

---

# 🧠 Methodology

VisionInspect follows a PatchCore-style unsupervised anomaly detection pipeline.

## 1. Normal-Only Learning

The system learns only from **normal training images**.

This is useful for industrial inspection because real manufacturing environments may have many possible defects while defect examples can be rare or unavailable.

---

## 2. Deep Feature Extraction

A pretrained:

**WideResNet50-2**

is used as the visual backbone.

Features are extracted from:

```text
layer2
layer3
```

For a `256 × 256` input image:

```text
layer2 → [512, 32, 32]
layer3 → [1024, 16, 16]
```

The `layer3` representation is resized to match the spatial resolution of `layer2`.

The two representations are concatenated:

```text
512 + 1024 = 1536 features
```

---

## 3. Local Patch Embeddings

The fused feature map contains:

```text
32 × 32 = 1024 local patches
```

Each patch initially has a:

```text
1536-dimensional feature vector
```

These embeddings represent local visual patterns such as shape, texture, edges and structure.

---

## 4. Feature Projection

The local representation is projected from:

```text
1536 dimensions
        ↓
128 dimensions
```

The projected embeddings are L2-normalized before nearest-neighbour comparison.

---

## 5. PatchCore Memory Bank

Normal training embeddings form the initial feature collection.

A smaller representative memory bank is created using a PatchCore-style coreset selection procedure.

Current configuration:

```text
Candidate patches: 10,000
Memory bank:       500
Embedding size:    128
```

Each category has its own memory bank:

```text
patchcore_bottle_memory.pt
patchcore_hazelnut_memory.pt
patchcore_tile_memory.pt
```

---

## 6. Nearest-Neighbour Anomaly Detection

Every test patch is compared against the category-specific normal memory bank.

Conceptually:

```text
Input Patch
    ↓
Deep Feature Embedding
    ↓
Nearest Normal Memory Patch
    ↓
Feature-Space Distance
    ↓
Patch Anomaly Score
```

A larger distance means the patch differs more strongly from the learned normal distribution.

---

## 7. Image-Level Anomaly Score

VisionInspect calculates a score for every local image patch.

The image-level score is based on the most anomalous local regions.

The current implementation uses approximately the **top 1% most anomalous patches**.

The decision rule is:

```text
score < threshold
        ↓
      NORMAL

score >= threshold
        ↓
    ANOMALOUS
```

---

# 🎯 Defect Localization

Patch anomaly scores form a spatial anomaly map.

The original patch grid is:

```text
32 × 32
```

It is interpolated to:

```text
256 × 256
```

and smoothed for visualization.

Warmer regions in the heatmap indicate stronger feature-space deviation from the normal memory bank.

---

# 🎬 Video Inspection Pipeline

Video inspection reuses the image anomaly-detection pipeline.

```text
Video
  ↓
Frame Sampling
  ↓
Image Preprocessing
  ↓
WideResNet50-2
  ↓
Patch Embeddings
  ↓
PatchCore Scoring
  ↓
Frame Anomaly Score
  ↓
Temporal Anomaly Profile
```

Sampling frames instead of analyzing every frame keeps CPU inference practical for the web demo.

---

# 📊 Development Evaluation

VisionInspect was evaluated on three MVTec AD categories.

| Category | Image AUROC | Pixel AUROC |
|---|---:|---:|
| Bottle | **1.0000** | **0.9819** |
| Hazelnut | **1.0000** | **0.9841** |
| Tile | **1.0000** | **0.9428** |
| **Average** | **1.0000** | **0.9696** |

Average F1 across the three evaluated categories:

```text
0.983
```

These results apply only to the three implemented categories and should **not** be interpreted as performance on the complete MVTec benchmark.

---

# ⚙️ Threshold Calibration

Initial percentile-based thresholds produced false positives for some normal images.

Additional validation-based threshold calibration was therefore performed.

Current deployment/demo thresholds:

| Category | Threshold | Calibration |
|---|---:|---|
| Bottle | **0.629979** | Normal validation mean + 3σ |
| Hazelnut | **0.774364** | Normal validation mean + 3σ |
| Tile | **0.669720** | Maximum normal validation score |

The threshold calculations themselves use normal validation data.

Test score distributions were also inspected during development for diagnostic purposes.

Therefore, classification results should be described as **development evaluation**, rather than performance on a completely untouched final test benchmark.

---

# 🧪 Baseline Comparison

Before PatchCore, a convolutional autoencoder was implemented as a reconstruction-based baseline.

Bottle results:

| Model | Image AUROC | Pixel AUROC |
|---|---:|---:|
| Convolutional Autoencoder | 0.8611 | 0.7184 |
| **PatchCore-style** | **1.0000** | **0.9819** |

The autoencoder detected some anomalies but produced noisier localization around normal bottle boundaries.

PatchCore-style deep feature matching produced substantially stronger image-level separation and localization.

---

# 🔬 Coreset Ablation Study

Different memory-bank sizes were evaluated on Bottle.

| Coreset Size | Image AUROC | Pixel AUROC | Inference ms/image | Memory |
|---:|---:|---:|---:|---:|
| 100 | 1.0000 | 0.9690 | 10.62 | 0.049 MB |
| 250 | 1.0000 | 0.9793 | 10.38 | 0.122 MB |
| **500** | **1.0000** | **0.9824** | **10.59** | **0.244 MB** |
| 1000 | 1.0000 | 0.9840 | 11.87 | 0.488 MB |

A **500-patch memory bank** was selected as the default.

Moving from 500 to 1000 patches approximately doubled memory usage while providing only a small improvement in pixel AUROC.

---

# 🗂️ Project Structure

```text
VisionInspect/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── docs/
│   ├── index.html
│   └── assets/
│       ├── bottle_anomaly.png
│       ├── hazelnut_normal.png
│       ├── tile_anomaly.png
│       ├── video_result.png
│       └── benchmark.png
│
├── src/
│   └── visioninspect/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── feature_extractor.py
│       ├── patchcore.py
│       ├── inference.py
│       └── video.py
│
├── models/
│   ├── autoencoder_bottle_best.pt
│   ├── patchcore_bottle_memory.pt
│   ├── patchcore_hazelnut_memory.pt
│   └── patchcore_tile_memory.pt
│
├── scripts/
│   ├── calibrate_bottle_threshold.py
│   ├── calibrate_remaining_categories.py
│   └── diagnose_bottle_threshold.py
│
├── notebooks/
│   └── VisionInspect_01_Bottle.ipynb
│
├── results/
│   ├── figures/
│   ├── heatmaps/
│   └── *.csv
│
├── data/
│
└── runtime/
```

The large dataset, virtual environment and runtime-generated files are excluded through `.gitignore`.

---

# 🚀 Local Installation

## 1. Clone the repository

```bash
git clone https://github.com/MuhammadTalha-pk/VisionInspect.git
cd VisionInspect
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the application

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 📦 Main Dependencies

VisionInspect uses:

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow
- OpenCV
- Matplotlib
- Pandas
- Scikit-learn
- Streamlit

---

# 💾 Dataset

The experiments use the **MVTec Anomaly Detection dataset**.

Expected structure:

```text
data/
├── bottle/
│   ├── train/
│   ├── test/
│   └── ground_truth/
│
├── hazelnut/
│   ├── train/
│   ├── test/
│   └── ground_truth/
│
└── tile/
    ├── train/
    ├── test/
    └── ground_truth/
```

The dataset itself is not included in this repository.

---

# ⚠️ Limitations

VisionInspect learns the visual distribution of each supported category.

It does not understand semantic object identity in the same way as a supervised classifier.

A real-world photograph of a bottle may therefore be considered anomalous if it differs substantially from the MVTec Bottle training distribution.

Potential domain-shift factors include:

- Camera viewpoint
- Lighting
- Background
- Product appearance
- Rotation
- Scale
- Texture
- Occlusion

External Bottle experiments demonstrated this sensitivity.

A production deployment would require normal reference images collected under the actual deployment camera and operating conditions.

---

# 🔬 What This Project Demonstrates

VisionInspect demonstrates practical experience with:

- Computer Vision
- Deep Learning
- Unsupervised Learning
- Industrial Visual Inspection
- Anomaly Detection
- PatchCore
- CNN Feature Extraction
- Transfer Learning
- Feature Embeddings
- Nearest-Neighbour Search
- Coreset Selection
- Threshold Calibration
- Pixel-Level Defect Localization
- Image Processing
- Video Processing
- Model Evaluation
- Ablation Studies
- Streamlit Deployment
- GitHub Pages

---

# 🔮 Future Work

Potential extensions include:

- Additional MVTec categories
- FAISS approximate nearest-neighbour search
- GPU inference
- Batch inspection
- Live camera inspection
- Improved temporal video aggregation
- Domain adaptation
- Few-shot adaptation
- Additional anomaly-detection architectures
- PaDiM comparison
- EfficientAD comparison
- Reverse Distillation comparison
- REST API deployment
- Docker support

---

# 📌 Project Links

**Live AI Demo**  
https://visioninspect-aidetection.streamlit.app

**Project Showcase**  
https://muhammadtalha-pk.github.io/VisionInspect/

**GitHub Repository**  
https://github.com/MuhammadTalha-pk/VisionInspect

---

## VisionInspect

**Unsupervised Industrial Defect Detection & Localization**

`PatchCore-style • WideResNet50-2 • Image + Video • Pixel-Level Localization`