import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Nano version for speed

# Streamlit page config
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🤖",
    layout="wide"
)

st.title("🚀 Advanced Object Detection with YOLOv8")
st.write("Upload an image, adjust thresholds, and detect objects in real time.")

# Sidebar settings
st.sidebar.header("⚙️ Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.01)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.01)

# File uploader
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Run YOLO detection
    results = model.predict(image_np, conf=conf_threshold, iou=iou_threshold)

    annotated_image = image_np.copy()
    detections = results[0]

    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])
        label = model.names[cls]

        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label text
        label_text = f"{label} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated_image, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(annotated_image, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # Display images
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_np, caption="📷 Original Image", use_column_width=True)
    with col2:
        st.image(annotated_image, caption="✅ Detected Objects", use_column_width=True)

else:
    st.info("👆 Upload an image to start detection.")

st.markdown("---")
st.caption("By Ashwik Bire | YOLOv8 + Streamlit 🚀")
