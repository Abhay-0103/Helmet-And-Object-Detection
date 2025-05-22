import cv2
from ultralytics import YOLO
from playsound import playsound
import threading
import os

# Load YOLOv8 model
model = YOLO('helmet.pt')  # Use your correct path here

# Mode: 'image', 'video', or 'webcam'
mode = 'webcam'

# Paths for image and video
image_path = ''
video_path = ''

# Beep control
beeping = False
beep_thread = None

# Absolute path for beep
BEEP_PATH = os.path.abspath("1100.wav")

# Function to play beep in loop
def play_beep():
    if not os.path.exists(BEEP_PATH):
        print("❌ beep.wav file not found!")
        return
    while beeping:
        playsound(BEEP_PATH)

# Detection logic
def process_frame(frame):
    global beeping, beep_thread
    results = model(frame)
    annotated = results[0].plot()
    names = model.names
    boxes = results[0].boxes
    detected_no_helmet = False

    for box in boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = names.get(cls_id, "unknown").lower()

        if class_name == "no helmet" and conf > 0.5:
            detected_no_helmet = True
            break

    if detected_no_helmet:
        cv2.putText(annotated, "No Helmet Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if not beeping:
            beeping = True
            beep_thread = threading.Thread(target=play_beep)
            beep_thread.start()
    else:
        cv2.putText(annotated, "Helmet Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if beeping:
            beeping = False
            if beep_thread and beep_thread.is_alive():
                beep_thread.join()

    return annotated

# Image detection
def detect_from_image(path):
    img = cv2.imread(path)
    annotated = process_frame(img)
    cv2.imshow('Helmet Detection - Image', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Video detection
def detect_from_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("❌ Failed to open video file.")
        return
    while True:
        success, frame = cap.read()
        if not success:
            break
        annotated = process_frame(frame)
        cv2.imshow('Helmet Detection - Video', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    stop_beep()
    cap.release()
    cv2.destroyAllWindows()

# Webcam detection
def detect_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return
    while True:
        success, frame = cap.read()
        if not success:
            break
        annotated = process_frame(frame)
        cv2.imshow('Helmet Detection - Webcam', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    stop_beep()
    cap.release()
    cv2.destroyAllWindows()

# Stop beep helper
def stop_beep():
    global beeping, beep_thread
    beeping = False
    if beep_thread and beep_thread.is_alive():
        beep_thread.join()

# ------------------------------
# Run detection
if mode == 'image':
    detect_from_image(image_path)
elif mode == 'video':
    detect_from_video(video_path)
elif mode == 'webcam':
    detect_from_webcam()
else:
    print("❌ Invalid mode. Choose from 'image', 'video', or 'webcam'.")
