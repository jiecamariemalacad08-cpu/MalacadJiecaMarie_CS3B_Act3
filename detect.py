import streamlit as st
import cv2
from ultralytics import YOLO
import os
import time
import csv
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Live Object Detection & Tracking", layout="wide")

st.title("🎥 Live Object Detection & Tracking using AI")
st.write("Real-time object detection + tracking using YOLOv8 + Streamlit")

# ---------------- Folder Setup ----------------
os.makedirs("saved_frames", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ---------------- Load YOLO ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")

target_object = st.sidebar.selectbox(
    "Choose object to count",
    ["person", "cell phone", "bottle", "chair", "laptop"]
)

# ---------------- Detection Log ----------------
log_file = "logs/detection_log.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Object", "Confidence", "Count"])

# ---------------- Video Processor ----------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.prev_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img, conf=0.5)
        object_count = 0

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])

                    if label == target_object:
                        object_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{label} {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

                    # Log
                    with open(log_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.now(), label, conf, object_count])

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time

        cv2.putText(img, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- Start Camera ----------------
st.subheader("📷 Live Camera")

webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

st.info("👉 Allow camera access when prompted by your browser.")
