# 🔍 VisionInspect

**Unsupervised Industrial Anomaly Detection & Defect Localization**

VisionInspect is a computer vision system for detecting and localizing visual defects in industrial products using a **PatchCore-style anomaly detection pipeline**.

The system learns from **normal images only** and identifies abnormal regions by comparing deep visual features against a memory bank of representative normal patterns.

The project currently supports three categories from the **MVTec Anomaly Detection dataset**:

- 🍾 Bottle
- 🌰 Hazelnut
- 🧱 Tile

VisionInspect provides both **image inspection** and **video inspection** through an interactive Streamlit application.

---

## ✨ Features

### Image Inspection

Upload an industrial image and receive:

- NORMAL / ANOMALOUS classification
- Image-level anomaly score
- Category-specific decision threshold
- Patch-level anomaly heatmap
- Defect localization overlay

### Video Inspection

Upload a video and VisionInspect can:

- Sample frames automatically
- Run anomaly detection on sampled frames
- Report anomalous-frame count
- Calculate anomaly rate
- Display maximum anomaly score
- Generate a temporal anomaly-score profile
- Show the most anomalous video frame
- Produce heatmap localization for the peak frame

### Interactive Dashboard

The Streamlit interface includes:

- Product-category selection
- Image / Video inspection modes
- Development benchmark metrics
- Architecture explanation
- Model configuration information
- Responsive dark industrial-AI interface

---

# 🧠 Methodology

VisionInspect follows a PatchCore-style unsupervised anomaly detection approach.

## 1. Normal-Only Learning

Only normal industrial training images are required.

The model does not need examples of every possible defect type.

## 2. Deep Feature Extraction

A pretrained:

**WideResNet50-2**

is used as the visual backbone.

Intermediate representations are extracted from:

```text
layer2
layer3
```

For a `256 × 256` image:

```text
layer2 → [512, 32, 32]
layer3 → [1024, 16, 16]
```

The `layer3` representation is resized to match the spatial resolution of `layer2`.

Both feature maps are then concatenated.

```text
512 + 1024 = 1536 features
```

---

## 3. Local Patch Embeddings

The fused feature representation is converted into local patch embeddings.

Each image produces:

```text
32 × 32 = 1024 local patches
```

Each patch initially contains a:

```text
1536-dimensional feature vector
```

The embeddings are L2-normalized.

---

## 4. Feature Projection

The original embeddings are projected from:

```text
1536 dimensions
        ↓
128 dimensions
```

This reduces memory requirements and improves nearest-neighbour computation efficiency.

The projected representations are normalized again before anomaly scoring.

---

## 5. PatchCore Memory Bank

Normal training patches form the initial feature collection.

A subset of candidate embeddings is retained and a PatchCore-style coreset is constructed.

The current configuration uses:

```text
Candidate patches: 10,000
Memory bank:       500
Embedding size:    128
```

Each supported category has its own memory bank:

```text
patchcore_bottle_memory.pt
patchcore_hazelnut_memory.pt
patchcore_tile_memory.pt
```

---

## 6. Nearest-Neighbour Anomaly Detection

For each test patch, VisionInspect calculates its distance from the normal memory bank.

The nearest normal embedding is identified.

A large distance indicates that the patch differs from learned normal visual patterns.

Conceptually:

```text
Test Patch
    ↓
Deep Embedding
    ↓
Nearest Normal Memory Patch
    ↓
Distance
    ↓
Patch Anomaly Score
```

---

## 7. Image-Level Anomaly Score

VisionInspect creates a spatial anomaly score for every patch.

The image-level score is calculated from the most anomalous local regions.

The current implementation uses the average score of approximately the **top 1% most anomalous patches**.

The final score is compared with a category-specific threshold.

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

The patch anomaly scores form a:

```text
32 × 32
```

anomaly map.

The map is interpolated to:

```text
256 × 256
```

and smoothed for visualization.

VisionInspect displays:

1. Original/preprocessed image
2. Anomaly heatmap
3. Localization overlay

Warmer heatmap regions represent stronger deviations from the learned normal feature distribution.

---

# 🎬 Video Inspection

Video inspection reuses the same image anomaly-detection pipeline.

Instead of analyzing every frame, VisionInspect can sample every Nth frame to keep CPU inference practical.

```text
Video
  ↓
Frame Sampling
  ↓
Image Preprocessing
  ↓
WideResNet Features
  ↓
PatchCore Scoring
  ↓
Frame Anomaly Score
  ↓
Temporal Anomaly Profile
```

The system reports:

- Number of analyzed frames
- Number of anomalous frames
- Maximum anomaly score
- Overall video inspection result
- Temporal anomaly-score graph
- Most anomalous frame
- Localization overlay for the peak frame

---

# 📊 Development Evaluation

VisionInspect was evaluated on three categories from the MVTec Anomaly Detection dataset.

| Category | Image AUROC | Pixel AUROC |
|---|---:|---:|
| Bottle | **1.0000** | **0.9819** |
| Hazelnut | **1.0000** | **0.9841** |
| Tile | **1.0000** | **0.9428** |
| **Average** | **1.0000** | **0.9696** |

These results apply only to the three evaluated categories and should **not** be interpreted as results for the complete MVTec benchmark.

---

# ⚙️ Threshold Calibration

Initial percentile-based thresholds produced false positives for some normal images.

Additional normal-validation calibration was therefore performed.

The deployment/demo thresholds currently used are:

| Category | Threshold | Calibration |
|---|---:|---|
| Bottle | **0.629979** | Normal validation mean + 3σ |
| Hazelnut | **0.774364** | Normal validation mean + 3σ |
| Tile | **0.669720** | Maximum normal validation score |

The threshold calculations themselves use normal validation data.

Test-score distributions were also inspected during development for diagnostic purposes. Therefore, classification results should be described as **development evaluation**, rather than as performance on a completely untouched final test benchmark.

---

# 🧪 Coreset Ablation

Bottle experiments were also performed with different memory-bank sizes.

| Coreset Size | Image AUROC | Pixel AUROC | Memory |
|---:|---:|---:|---:|
| 100 | 1.0000 | 0.9690 | ~0.049 MB |
| 250 | 1.0000 | 0.9793 | ~0.122 MB |
| 500 | 1.0000 | 0.9824 | ~0.244 MB |
| 1000 | 1.0000 | 0.9840 | ~0.488 MB |

A coreset size of **500** was selected as the default because it offers a strong trade-off between localization performance, memory usage, and inference cost.

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

The `data/`, `.venv/`, and runtime-generated files are excluded from GitHub through `.gitignore`.

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone <repository-url>
cd VisionInspect
```

## 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running VisionInspect

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

The application supports both:

```text
Image Inspection
Video Inspection
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

The experiments use the **MVTec Anomaly Detection (MVTec AD)** dataset.

Expected directory structure:

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

VisionInspect is trained on controlled MVTec category distributions.

Its anomaly detector does not understand the semantic concept of an object in the same way as a supervised classifier.

For example, a real-world photograph of a bottle may still be considered anomalous if it differs substantially from the MVTec Bottle training distribution.

Changes such as:

- Camera viewpoint
- Lighting
- Background
- Product design
- Scale
- Rotation
- Texture
- Occlusion

can produce **domain-shift false positives**.

This behavior is expected from nearest-neighbour anomaly detection when the deployment distribution differs substantially from the normal reference distribution.

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
- Pixel-Level Localization
- Image Processing
- Video Processing
- Model Evaluation
- Ablation Studies
- Streamlit Deployment

---

# 🔮 Future Work

Potential improvements include:

- Additional MVTec categories
- Approximate nearest-neighbour search using FAISS
- GPU inference
- Batch inspection
- Real-time camera inspection
- Improved video temporal aggregation
- Domain adaptation
- Few-shot adaptation to new production environments
- Alternative pretrained backbones
- PaDiM / EfficientAD / Reverse Distillation comparisons
- REST API deployment
- Docker support

---

# 📌 Research Note

The project is designed as an educational and portfolio implementation of industrial visual anomaly detection.

It is not intended to represent a production-certified quality-control system without further validation on deployment-specific industrial data.

---

## VisionInspect

**Deep visual anomaly detection for industrial inspection.**

`PatchCore-style • WideResNet50-2 • Image + Video • Pixel-Level Localization`