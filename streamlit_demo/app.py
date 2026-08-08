import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ---------------------------------------------------------
# Safety Helmet Detection Demo
# AIRI Team PITB - AI Internship Task 1 - Phase 11 Demo
# ---------------------------------------------------------

st.set_page_config(page_title="Safety Helmet Detection", layout="centered")

st.title("🪖 Safety Helmet Detection System")
st.write(
    "Upload an image and the model will detect **Hardhat**, **NO-Hardhat**, "
    "and **Person** with bounding boxes and confidence scores."
)

# ---------------------------------------------------------
# Load model (cached so it only loads once per session)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    # Update this path to wherever best.pt lives on the machine
    # running this app (local copy, or a Drive-mounted path).
    return YOLO("best.pt")

model = load_model()

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider(
    "Confidence threshold", min_value=0.05, max_value=0.95, value=0.35, step=0.05
)

# ---------------------------------------------------------
# Upload + inference
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    with st.spinner("Running detection..."):
        results = model.predict(
            source=np.array(image),
            conf=conf_threshold,
            save=False,
        )

    result = results[0]
    annotated = result.plot()  # returns a numpy array (BGR) with boxes drawn

    with col2:
        st.subheader("Detections")
        st.image(annotated, channels="BGR", use_container_width=True)

    # ---------------------------------------------------------
    # Detection summary table
    # ---------------------------------------------------------
    st.subheader("Detection Details")
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        rows = []
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            rows.append({"Class": cls_name, "Confidence": f"{conf:.2f}"})
        st.table(rows)
    else:
        st.info("No objects detected above the confidence threshold.")

else:
    st.info("Upload an image to get started.")

st.markdown("---")
st.caption("Built with YOLOv8 (Ultralytics) — AIRI Team PITB AI Internship Task 1")
