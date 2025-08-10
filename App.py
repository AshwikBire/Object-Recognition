import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Object Detection (Dark Mode)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode tweaks
st.markdown("""
    <style>
    body { background: #111 !important; color: #ddd !important; }
    .block-container { background: #18181b !important; }
    [data-testid="stSidebar"] { background: #191926 !important; color: #e8e8e8 !important; }
    .stSlider > div > div { background: #24242d !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🦾 Object Detection — YOLOv8 (Dark Mode Enhanced)")

# Sidebar controls
st.sidebar.header("Input Controls")

# Options: File upload or camera capture
input_mode = st.sidebar.radio(
    "Select Input Source:",
    ("Upload Image/Video", "Capture Image With Camera"),
    key="input_mode"
)

confidence = st.sidebar.slider("Detection Confidence", 0.25, 0.99, 0.5, 0.01)
show_labels = st.sidebar.checkbox("Show labels", value=True)
show_scores = st.sidebar.checkbox("Show scores", value=True)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

def run_detection(img, conf):
    results = model(img, conf=conf)
    return results

def draw_boxes(results, show_labels=True, show_scores=True):
    import cv2
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    labels = results.names
    scores = results.boxes.conf.cpu().numpy()
    img_bbox = results.orig_img.copy()
    for box, cls, score in zip(boxes, results.boxes.cls.cpu().numpy().astype(int), scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
        txt = labels[cls] if show_labels else ""
        if show_scores:
            txt += f" {score:.2f}"
        if txt:
            cv2.putText(img_bbox, txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 180), 2)
    return Image.fromarray(img_bbox)

# Input handling
if input_mode == "Upload Image/Video":
    uploaded_file = st.sidebar.file_uploader("Upload image (JPG, PNG) or video (MP4)", type=["jpg", "jpeg", "png", "mp4"])
    if uploaded_file:
        if uploaded_file.type.startswith("image"):
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Original Image", width=480)
            results = run_detection(np.array(img), confidence)
            img_annot = draw_boxes(results[0], show_labels, show_scores)
            st.image(img_annot, caption="Detected Objects", width=480)
            st.download_button("Download Result Image", data=img_annot.tobytes(), mime="image/png")
            st.subheader("Detection Data")
            st.dataframe(results[0].pandas().xyxy[0])
        elif uploaded_file.type.startswith("video"):
            st.video(uploaded_file)
            st.info("To process video frames for detection, run locally due to compute constraints. Try uploading an image for best demo!")
    else:
        st.markdown("⬅️ Please upload an image or video file to start detection.")
elif input_mode == "Capture Image With Camera":
    captured_image = st.camera_input("Take a photo", key="camera")
    if captured_image:
        img = Image.open(captured_image).convert("RGB")
        st.image(img, caption="Captured Image", width=480)
        results = run_detection(np.array(img), confidence)
        img_annot = draw_boxes(results[0], show_labels, show_scores)
        st.image(img_annot, caption="Detected Objects", width=480)
        st.download_button("Download Result Image", data=img_annot.tobytes(), mime="image/png")
        st.subheader("Detection Data")
        st.dataframe(results[0].pandas().xyxy[0])
    else:
        st.markdown("📸 Use your webcam/camera to take a picture for real-time object detection.")

st.markdown("---")
st.markdown(
    "<center style='color:#ccc'>Built with <b>Streamlit</b> & <b>YOLOv8</b> • Supports dark theme & live image capture</center>",
    unsafe_allow_html=True
)
