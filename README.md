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
```

## 🤖 Model Training

The project uses **YOLOv8n pretrained weights** as the starting point.

| Setting    | Value        |
| ---------- | ------------ |
| Model      | YOLOv8n      |
| Epochs     | 30           |
| Image Size | 640          |
| Batch Size | 8            |
| Platform   | Google Colab |
| GPU        | Tesla T4     |

The best model checkpoint was saved as `best.pt`.

## 📈 Evaluation Results

| Metric       | Score |
| ------------ | ----: |
| Precision    | 0.598 |
| Recall       | 0.473 |
| mAP@0.5      | 0.476 |
| mAP@0.5:0.95 | 0.260 |

### Class Performance

| Class      | mAP@0.5 |
| ---------- | ------: |
| Hardhat    |   0.707 |
| Person     |   0.622 |
| NO-Hardhat |   0.100 |

Hardhat was the best-performing class, while NO-Hardhat was the weakest class. The main reason identified was class imbalance, with substantially fewer NO-Hardhat training examples.

## 🔍 Error Analysis

The model was manually reviewed on held-out test images.

Identified error types included:

* False positives
* False negatives
* Wrong or missed class predictions
* Poor bounding boxes
* Small-object detection issues
* Low-confidence detections
* Problems with overlapping objects

Performance was generally better for clear and well-lit scenes and weaker for small/distant people and the NO-Hardhat class.

## 🌐 Streamlit Demo

A Streamlit interface was developed to run inference using the trained YOLOv8 model.

The application allows an image to be provided as input and displays the detected objects with their bounding boxes, class labels, and confidence scores.

## 📁 Suggested Project Structure

```text
Safety-Helmet-Detection/
│
├── README.md
├── .gitignore
├── app.py
├── requirements.txt
│
├── model/
│   └── best.pt
│
├── notebooks/
│   └── training.ipynb
│
├── images/
│   └── sample.jpg
│
└── src/
    └── ...
```

Adjust the structure according to the actual files in the project.

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Safety-Helmet-Detection.git
cd Safety-Helmet-Detection
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Streamlit Demo

```bash
streamlit run app.py
```

The Streamlit application can then be opened in a browser to perform inference on an image.

## 🔮 Future Improvements

* Add more training images, especially for the NO-Hardhat class
* Improve class balance
* Add low-light, blurry, and occluded examples
* Train a larger YOLO model such as YOLOv8s or YOLOv8m
* Improve annotation quality
* Deploy the application as a public web app
* Test the model on real-time video and CCTV footage

## 📚 Learning Outcomes

Through this project, I learned how to:

* Prepare and organize an object-detection dataset
* Perform manual bounding-box annotation
* Export datasets in YOLO format
* Identify dataset issues and data leakage
* Train YOLOv8 using Google Colab
* Evaluate object-detection models using precision, recall, and mAP
* Perform structured error analysis
* Build an interactive inference demo
