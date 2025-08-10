# app.py

import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Advanced Object Detection", layout="wide")

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  # Use YOLOv8n for speed, or change to yolov8s.pt, etc.

st.title("🔎 Advanced Object Detection with YOLOv8 and Streamlit")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background: #fafafa }
    .css-1aumxhk { font-size:22px !important; color:#0066cc; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Input Options")
uploaded_file = st.sidebar.file_uploader(
    "Upload an image (JPG, PNG) or video (MP4)", type=["png", "jpg", "jpeg", "mp4"]
)
confidence = st.sidebar.slider("Detection Confidence", 0.25, 0.99, 0.5, 0.01)
show_labels = st.sidebar.checkbox("Show labels", value=True)
show_scores = st.sidebar.checkbox("Show scores", value=True)

model = load_model()

def run_detection(img, conf):
    results = model(img, conf=conf)
    return results

def draw_boxes(results, show_labels=True, show_scores=True):
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    labels = results.names
    scores = results.boxes.conf.cpu().numpy()
    img_bbox = results.orig_img.copy()
    from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX
    for box, cls, score in zip(boxes, results.boxes.cls.cpu().numpy().astype(int), scores):
        x1, y1, x2, y2 = box
        rectangle(img_bbox, (x1, y1), (x2, y2), (0,255,0), 2)
        txt = labels[cls] if show_labels else ""
        if show_scores:
            txt += f" {score:.2f}"
        if txt:
            putText(img_bbox, txt, (x1, y1-10), FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return Image.fromarray(img_bbox)

if uploaded_file:
    if uploaded_file.type.startswith("image"):
        st.image(uploaded_file, caption="Original Image", width=480)
        img = Image.open(uploaded_file).convert("RGB")
        results = run_detection(np.array(img), confidence)
        img_annot = draw_boxes(results[0], show_labels, show_scores)
        st.image(img_annot, caption="Detected Objects", width=480)
        st.download_button(
            "Download Result Image", data=img_annot.tobytes(), mime="image/png"
        )
        st.subheader("Detection Data")
        st.dataframe(results[0].pandas().xyxy[0])
    elif uploaded_file.type.startswith("video"):
        st.video(uploaded_file)
        st.info(
            "Video detection is best run locally due to compute. Try uploading an image!"
        )
else:
    st.info("Upload a JPG/PNG image or MP4 video to run detection.")

st.markdown("---")
st.markdown("<center>🦾 Built with Streamlit & YOLOv8 | <a href='https://ultralytics.com/yolov8/' target='_blank'>YOLOv8 Docs</a></center>", unsafe_allow_html=True)
