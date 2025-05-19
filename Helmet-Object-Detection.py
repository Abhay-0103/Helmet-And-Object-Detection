import cv2
from ultralytics import YOLO
import numpy as np

# Load models
general_model = YOLO('yolov8n.pt')   # General model (e.g., person, car, etc.)
helmet_model = YOLO('helmet.pt')     # Custom helmet detection model

mode = 'webcam'  # Change to 'image' or 'video' if needed
image_path = ''
video_path = ''

def draw_boxes(frame, boxes, names, label_filter=None, conf_threshold=0.5, box_color=(0,255,0)):
    """Draw bounding boxes on the frame filtered by label_filter and confidence."""
    for box in boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = names.get(cls_id, "unknown").lower()
        if conf < conf_threshold:
            continue
        if label_filter and label_filter not in class_name:
            continue
        
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

def detect_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run general detection
        results_general = general_model(frame)
        annotated_frame = results_general[0].plot()  # Annotated with general model detections
        
        # Run helmet detection on original frame
        results_helmet = helmet_model(frame)
        
        # Draw helmet boxes manually on annotated_frame
        draw_boxes(annotated_frame, results_helmet[0].boxes, helmet_model.names, label_filter="helmet", conf_threshold=0.5, box_color=(0,255,0))
        
        # Decide helmet status based on helmet detections
        helmet_detected = any(
            ("helmet" in helmet_model.names[int(box.cls[0])].lower() and float(box.conf[0]) > 0.5)
            for box in results_helmet[0].boxes
        )

        if helmet_detected:
            cv2.putText(annotated_frame, "Helmet Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_frame, "No Helmet Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Helmet & Object Detection - Webcam', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_from_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Failed to read image from {path}")
        return

    results_general = general_model(img)
    annotated = results_general[0].plot()
    
    results_helmet = helmet_model(img)
    draw_boxes(annotated, results_helmet[0].boxes, helmet_model.names, label_filter="helmet", conf_threshold=0.5, box_color=(0,255,0))
    
    helmet_detected = any(
        ("helmet" in helmet_model.names[int(box.cls[0])].lower() and float(box.conf[0]) > 0.5)
        for box in results_helmet[0].boxes
    )
    if helmet_detected:
        cv2.putText(annotated, "Helmet Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(annotated, "No Helmet Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Helmet & Object Detection - Image', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_from_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"❌ Failed to open video file: {path}")
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        results_general = general_model(frame)
        annotated_frame = results_general[0].plot()
        
        results_helmet = helmet_model(frame)
        draw_boxes(annotated_frame, results_helmet[0].boxes, helmet_model.names, label_filter="helmet", conf_threshold=0.5, box_color=(0,255,0))

        helmet_detected = any(
            ("helmet" in helmet_model.names[int(box.cls[0])].lower() and float(box.conf[0]) > 0.5)
            for box in results_helmet[0].boxes
        )

        if helmet_detected:
            cv2.putText(annotated_frame, "Helmet Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_frame, "No Helmet Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Helmet & Object Detection - Video', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if mode == 'image':
        detect_from_image(image_path)
    elif mode == 'video':
        detect_from_video(video_path)
    elif mode == 'webcam':
        detect_from_webcam()
    else:
        print("❌ Invalid mode. Choose from 'image', 'video', or 'webcam'.")
