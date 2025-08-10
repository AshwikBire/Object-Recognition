import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model for object detection
model = YOLO("yolov8n.pt")  # yolov8 nano for speed/balance, replace with larger model for accuracy

def detect_objects(image, conf_threshold, iou_threshold):
    """
    Perform object detection on the input image using YOLOv8.

    Args:
        image (np.array): Input RGB image as numpy array
        conf_threshold (float): Confidence threshold for detections
        iou_threshold (float): IOU threshold for non-max suppression

    Returns:
        np.array: Image with bounding boxes and labels drawn
    """
    # Run detection
    results = model.predict(image, conf=conf_threshold, iou=iou_threshold)

    # Copy image to annotate
    annotated_image = image.copy()

    # Extract detection results for the first image only (batch size=1)
    detections = results[0]
    
    # Draw each detection box and label on the image
    for box in detections.boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])
        label = model.names[cls]

        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Prepare label text with confidence
        label_text = f"{label} {conf:.2f}"

        # Calculate label position
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated_image, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)

        # Put label over the bounding box
        cv2.putText(annotated_image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated_image

# Gradio Interface Setup  
title = "Advanced Object Detection with YOLOv8 and Python"
description = "Upload an image, adjust confidence and IOU thresholds, and detect objects with a fast YOLOv8 model."

iface = gr.Interface(
    fn=detect_objects,
    inputs=[
        gr.Image(type="numpy", label="Input Image"),
        gr.Slider(0, 1, value=0.3, step=0.01, label="Confidence Threshold"),
        gr.Slider(0, 1, value=0.45, step=0.01, label="IOU Threshold")
    ],
    outputs=gr.Image(type="numpy", label="Detected Objects"),
    title=title,
    description=description,
    examples=[
        ["example1.jpg", 0.3, 0.45],  # You can supply example images in your folder
        ["example2.jpg", 0.25, 0.4]
    ],
    theme="default"
)

if __name__ == "__main__":
    iface.launch()
