# Multimodal Fall Detection System (UP-Fall Dataset)

This repository contains the implementation of a Fall Detection System
(FDS) based on Hybrid Data Fusion. The project investigates the
effectiveness of combining Computer Vision (RGB images) and Wearable
Sensors (accelerometer data) to improve detection accuracy and
reliability compared to single-modality approaches.

**Project Overview** Falls are a leading cause of injury and
mortality among the elderly population. Traditional systems often suffer
from:

-   **Sensor-only limitations**: High false alarm rates (e.g., confusing
    sitting down with falling).
-   **Vision-only limitations**: Privacy concerns, occlusion (objects
    blocking the view), and lighting conditions.

This project addresses these issues by implementing and comparing three
fusion strategies:

1.  **Early Fusion (Feature-level)** --- Combining extracted features
    from both modalities into one feature vector.
2.  **Late Fusion (Decision-level)** --- Averaging probability scores of
    independent models.
3.  **Logical Fusion (Rule-based)** --- A logical OR rule to maximize
    Recall.

**System Architecture**

### 1. Vision Branch (CNN)

-   **Input:** RGB Images (64×64 pixels)
-   **Backbone:** MobileNetV2 (Transfer Learning from ImageNet)
-   **Function:** Extracts spatial features describing posture.

### 2. Sensor Branch (RNN)

-   **Input:** Accelerometer time-series window (20×3)
-   **Backbone:** 1D-CNN + LSTM
-   **Function:** Detects sudden acceleration changes typical for falls.

**Dataset Structure** The project uses the **UP-Fall Detection
Dataset**.\
Subject used: **Subject 1**

### Required Folder Structure:

    Project_Root/
    ├── Dataset/
    │   ├── Subject1.0/
    │   │   ├── Activity1.0/
    │   │   │   └── Trial1.0/
    │   │   │       ├── Camera1/   # .png images
    │   │   │       └── *.csv      # Sensor data
    │   │   ├── Activity6.0/
    │   │   ...
    ├── model.ipynb
    └── README.md

**Installation and Usage**

### Prerequisites

-   Python 3.8+

### Clone the repository:

``` bash
git clone <your-repo-link>
cd <your-project-folder>
```

### Install dependencies:

``` bash
pip install tensorflow pandas numpy opencv-python matplotlib scikit-learn seaborn
```

### Run the project:

``` bash
jupyter notebook model.ipynb
```

📊 **Experimental Results**

  ------------------------------------------------------------------------
  Strategy         Accuracy   Precision(Fall)    Recall(Fall)   F1-Score
  ---------------- ---------- ------------------ -------------- ----------
  Early Fusion     99.8%      1.00               1.00           1.00

  Late Fusion      87.2%      0.85               0.78           0.81

  Logical Fusion   97.9%      0.90               0.84           0.87
  ------------------------------------------------------------------------

### Key Findings

-   **Early Fusion** performed the best, achieving near-perfect
    accuracy.
-   **Late Fusion** behaved like a "lazy predictor" because models
    worked independently.
-   **Logical Fusion** provided high Recall → good for safety
    applications.

🔮 **Future Work** - Edge deployment (Raspberry Pi 4) -
Privacy-preserving skeleton-based vision pipeline - Training on all 17
subjects for better generalization
