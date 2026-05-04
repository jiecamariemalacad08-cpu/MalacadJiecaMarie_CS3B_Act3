import streamlit as st
import cv2
from ultralytics import YOLO
import os
import time
import csv
from datetime import datetime

st.set_page_config(
    page_title="Live Object Detection & Tracking",
    layout="wide"
)

st.title("🎥 Live Object Detection & Tracking using AI")
st.write("Real-time object detection + tracking using YOLOv8 + OpenCV + Streamlit")

if not os.path.exists("saved_frames"):
    os.makedirs("saved_frames")

if not os.path.exists("logs"):
    os.makedirs("logs")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.sidebar.header("Settings")

target_object = st.sidebar.selectbox(
    "Choose object to count",
    ["person", "cell phone", "bottle", "chair", "laptop"]
)

run = st.sidebar.checkbox("Start Camera")
save_frame = st.sidebar.button("Save Current Frame")

FRAME_WINDOW = st.image([])
count_display = st.empty()
fps_display = st.sidebar.empty()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

log_file = "logs/detection_log.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Object", "Confidence", "Count"])

if run:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error("❌ Camera not detected. Please check webcam permissions.")
        st.stop()

    out = cv2.VideoWriter('logs/output.avi', fourcc, 20.0, (640, 480))

    prev_time = time.time()

    while run:
        success, frame = cap.read()
        if not success:
            st.error("❌ Failed to read from camera")
            break

        results = model(frame, conf=0.5)
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
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                
                    cv2.putText(frame, f"{label} {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)

                    with open(log_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.now(), label, conf, object_count])

        count_display.success(f"Detected {target_object}: {object_count}")

        if target_object == "cell phone" and object_count > 0:
            st.warning("📱 Cell phone detected!")

        if target_object == "person" and object_count > 5:
            st.error("⚠ Too many people detected!")

        if save_frame:
            filename = datetime.now().strftime("saved_frames/frame_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(filename, frame)
            st.success(f"✅ Frame saved: {filename}")

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_display.write(f"FPS: {fps:.2f}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

        out.write(frame)

    cap.release()
    out.release()

else:
    st.info("☝ Click 'Start Camera' in the sidebar to begin detection.")
