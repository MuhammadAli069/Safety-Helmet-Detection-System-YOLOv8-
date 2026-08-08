# Safety Helmet Detection — Streamlit Demo

Simple web app: upload an image, get back bounding boxes for **Hardhat**,
**NO-Hardhat**, and **Person**, with confidence scores.

## How to run this locally

1. Download `best.pt` from your Drive (`cv_project/models/best.pt`) and place
   it in this same folder, next to `app.py`.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Your browser will open automatically at `http://localhost:8501`.
   Upload any image and see live detections.

## Notes
- Adjust the confidence threshold slider in the sidebar to see how detections
  change.
- This satisfies Phase 11 (Option A: Streamlit App) of the AIRI Team PITB
  AI Internship Task 1.
