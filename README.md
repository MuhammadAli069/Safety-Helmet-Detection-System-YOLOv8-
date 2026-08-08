# 🦺 Safety Helmet Detection System using YOLOv8

A Computer Vision object-detection system built with **YOLOv8** to detect whether workers on a construction site are wearing safety helmets.

The model detects three classes:

* **Hardhat**
* **NO-Hardhat**
* **Person**

The system takes an image as input and produces bounding boxes, predicted class labels, and confidence scores for detected objects.

## 🚀 Features

* Detects workers and helmet compliance
* Identifies **Hardhat**, **NO-Hardhat**, and **Person**
* Provides bounding boxes and confidence scores
* Trained using YOLOv8
* Includes evaluation and error analysis
* Streamlit-based inference demo
* Can be extended to real-time video/CCTV detection

## 🛠️ Technologies Used

* **Python**
* **YOLOv8 (Ultralytics)**
* **Google Colab**
* **Tesla T4 GPU**
* **Google Drive**
* **Roboflow**
* **OpenCV**
* **Pillow**
* **Streamlit**

## 📊 Dataset

The dataset contains **200 images** across the three detection classes.

| Split      |  Images | Percentage |
| ---------- | ------: | ---------: |
| Training   |     140 |        70% |
| Validation |      40 |        20% |
| Test       |      20 |        10% |
| **Total**  | **200** |   **100%** |

The dataset was prepared using Roboflow, including manual annotation and data cleaning. Near-duplicate images and problematic bounding boxes were removed during preparation.

### Classes

```text
0 - Hardhat
1 - NO-Hardhat
2 - Person