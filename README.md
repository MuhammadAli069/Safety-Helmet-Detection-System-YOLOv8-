# 🦺 Safety Helmet Detection System using YOLOv8

A Computer Vision object-detection system built with **YOLOv8** to detect whether workers on a construction site are wearing safety helmets.

> AIRI Team PITB — AI Internship Task 1
> Prepared by **Syed Muhammad Ali Kazmi** · Submitted to Omar Farooq, Associate AI/ML Engineer

The model detects three classes:

* **Hardhat**
* **NO-Hardhat**
* **Person**

The system takes an image (or video) as input and produces bounding boxes, predicted class labels, and confidence scores for every detected object.

📄 Full project report: [`report/final_report.pdf`](report/final_report.pdf)

## 🚀 Features

* Detects workers and helmet compliance
* Identifies **Hardhat**, **NO-Hardhat**, and **Person**
* Provides bounding boxes and confidence scores
* Trained using YOLOv8n on Google Colab (Tesla T4 GPU)
* Includes full evaluation (precision, recall, mAP, confusion matrix) and error analysis
* Streamlit-based inference demo (upload an image, get annotated results)
* Video inference support, extendable to real-time CCTV feeds

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

The dataset contains **200 images** across the three detection classes, combining a manually re-annotated split with a cleaned subset of the [Construction Site Safety Object Detection](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/30) dataset (Roboflow Universe, CC BY 4.0).

| Split      |  Images | Percentage |
| ---------- | ------: | ---------: |
| Training   |     140 |        70% |
| Validation |      40 |        20% |
| Test       |      20 |        10% |
| **Total**  | **200** |   **100%** |

The training split was manually annotated in Roboflow. Data cleaning included converting polygon exports to YOLO bounding-box format, removing near-duplicate images that leaked across splits (via perceptual hashing), and dropping a few degenerate (near-zero-area) boxes.

### Classes

```text
0 - Hardhat
1 - NO-Hardhat
2 - Person
```

## 🏋️ Model Training

| Setting       | Value                |
| ------------- | -------------------- |
| Model         | YOLOv8n (pretrained) |
| Epochs        | 30                   |
| Image size    | 640                  |
| Batch size    | 8                    |
| Platform      | Google Colab         |
| GPU           | Tesla T4 (14913 MiB) |
| Training time | ~6 minutes           |

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data="/content/drive/MyDrive/cv_project/dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    pretrained=True,
    project="/content/drive/MyDrive/cv_project/outputs/training_results",
    name="yolov8n_baseline",
)
```

## 📈 Evaluation Results

| Metric       | Value |
| ------------ | ----: |
| Precision    | 0.598 |
| Recall       | 0.473 |
| mAP@0.5      | 0.476 |
| mAP@0.5:0.95 | 0.260 |

| Class              | mAP@0.5 |
| ------------------ | ------: |
| Hardhat (best)     |   0.707 |
| Person             |   0.622 |
| NO-Hardhat (worst) |   0.100 |

NO-Hardhat is the clear weak point, driven mainly by class imbalance in the training data (far fewer NO-Hardhat examples than Hardhat or Person). Full confusion matrices and loss curves are in the report.

## 🔍 Inference

The trained model was run on all 20 held-out test images (confidence threshold 0.35), plus a short real-world construction video clip for extended video inference. Sample predictions and the full 20-image grid are in the report and in `outputs/predictions/`.

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
results = model.predict(
    source="dataset/images/test",
    conf=0.35,
    save=True,
    project="outputs/predictions",
    name="test_predictions",
)
```

## 🐞 Error Analysis

At least 10 predictions were manually reviewed and logged (see the report for the full table). Recurring issues:

* False positives on cluttered backgrounds / ambiguous shapes
* Missed small or distant people
* Weak NO-Hardhat recognition (class imbalance)
* Loose/overlapping bounding boxes in dense, crowded scenes

**Suggested improvements:** add more NO-Hardhat images, train longer or try YOLOv8s/YOLOv8m, add more low-light and motion-blur examples, balance the dataset, and tighten annotation quality.

## 🖥️ Demo

A Streamlit app is included for interactive inference — upload an image and get back the annotated result plus a detection-details table.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Repository Structure

```text
cv_project/
├── dataset/
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── data.yaml
├── notebooks/
│   └── training_notebook.ipynb
├── outputs/
│   ├── training_results/
│   ├── predictions/
│   └── demo_results/
├── models/
│   └── best.pt
├── report/
│   └── final_report.pdf
├── README.md
└── requirements.txt
```

## ▶️ Getting Started

```bash
git clone https://github.com/MuhammadAli069/Safety-Helmet-Detection-System-YOLOv8-.git
cd Safety-Helmet-Detection-System-YOLOv8-
pip install -r requirements.txt
```

Run inference with your own image:

```bash
python -c "
from ultralytics import YOLO
model = YOLO('models/best.pt')
model.predict(source='path/to/image.jpg', conf=0.35, save=True)
"
```

## 🎓 What I Learned

* How to prepare and organize a Computer Vision dataset for object detection
* Manual bounding-box annotation and exporting to YOLO format
* Diagnosing and fixing dataset issues (label format mismatches, split leakage)
* Training YOLOv8 end-to-end on Google Colab
* Evaluating object detectors with precision, recall, and mAP
* Structured error analysis on model predictions
* Building a simple interactive demo around a trained model

## 🔮 Future Improvements

* Add more images, especially for the NO-Hardhat class
* Improve class balance across the dataset
* Add more low-light, occluded, and small-object examples
* Train a larger YOLO model (YOLOv8s / YOLOv8m)
* Improve annotation quality and consistency
* Deploy the demo as a public web app
* Test on real-time video / CCTV footage

## 👤 Author

**Syed Muhammad Ali Kazmi**
AIRI Team PITB — AI Internship Task 1

## 📄 License

Dataset used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) via Roboflow Universe. Code in this repository is provided for educational/portfolio purposes.