import cv2
from ultralytics import YOLO

# Load YOLOv8 model (use your custom helmet detection model)
model = YOLO('helmet.pt')  # Change to your model path if needed

# Mode: 'image', 'video', or 'webcam'
mode = 'webcam'

# Paths for image and video (if using those modes)
image_path = ''
video_path = ''

# ------------------------------
# Function: Image detection
def detect_from_image(path):
    img = cv2.imread(path)
    results = model(img)
    annotated = results[0].plot()
    cv2.imshow('Helmet Detection - Image', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------------------------------
# Function: Video detection
def detect_from_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("❌ Failed to open video file.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        names = model.names
        boxes = results[0].boxes
        labels = []

        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = names.get(cls_id, "unknown")

            # Accept only confident helmet detections
            if "helmet" in class_name.lower() and conf > 0.5:
                labels.append(class_name)

        if labels:
            cv2.putText(annotated_frame, "Helmet Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_frame, "No Helmet Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Helmet Detection - Video', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------------------
# Function: Webcam detection
def detect_from_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        names = model.names
        boxes = results[0].boxes
        labels = []

        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = names.get(cls_id, "unknown")

            # Accept only confident helmet detections
            if "helmet" in class_name.lower() and conf > 0.5:
                labels.append(class_name)

        if labels:
            cv2.putText(annotated_frame, "Helmet Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_frame, "No Helmet Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Helmet Detection - Webcam', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------------------
# Run detection based on mode
if mode == 'image':
    detect_from_image(image_path)
elif mode == 'video':
    detect_from_video(video_path)
elif mode == 'webcam':
    detect_from_webcam()
else:
    print("❌ Invalid mode. Choose from 'image', 'video', or 'webcam'.")
