import cv2
from ultralytics import YOLO
import numpy as np

# YOLO model and settings
model_path = "yolov8l.pt"
target_labels = ["cell phone", "bottle"]  # Labels you want to detect
model = YOLO(model_path)

# OpenCV video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error accessing the camera.")
    exit()

# Reduce frame size for better performance
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize variables
prev_gray = None
tracked_objects = {}

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y


def detect_and_track(frame):
    global prev_gray, tracked_objects

    centers = {}
    yolo_frame = frame.copy()
    optical_flow_frame = frame.copy()
    combined_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = model(frame, conf=0.7, verbose=False)  # Increase confidence threshold
    detections = []

    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0].item()

            # Filter by confidence and desired class
            if label in target_labels and confidence > 0.7:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x, center_y = get_center((x1, y1, x2, y2))
                detections.append((center_x, center_y, label))
                cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(yolo_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                centers[label] = center_y

                # Add y-coordinate text above the bounding box
                cv2.putText(yolo_frame, f"y: {center_y}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if prev_gray is None:
        prev_gray = gray
        for (center_x, center_y, label) in detections:
            tracked_objects[label] = (center_x, center_y)
    else:
        if tracked_objects:
            y_offset = 30  # Initial y offset for the text
            for label, point in tracked_objects.items():
                prev_points = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
                good_new = next_points[status == 1]
                good_old = prev_points[status == 1]

                if len(good_new) > 0:
                    new = good_new[0]
                    old = good_old[0]
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a, b = int(a), int(b)
                    c, d = int(c), int(d)
                    tracked_objects[label] = (a, b)
                    cv2.circle(optical_flow_frame, (a, b), 5, (0, 255, 0), -1)
                    cv2.line(optical_flow_frame, (a, b), (c, d), (0, 255, 0), 2)

                    # Print the y-coordinate of the optical flow
                    print(f"Optical flow y-coordinate for {label}: {b}")
                    centers[label] = b

                    # Add y-coordinate text in the corner of the optical flow frame
                    cv2.putText(optical_flow_frame, f"y: {b}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    y_offset += 30  # Increment y offset for the next text

        for (center_x, center_y, label) in detections:
            tracked_objects[label] = (center_x, center_y)

        prev_gray = gray.copy()

    # Combine YOLO and Optical Flow results in the combined frame
    combined_frame = cv2.addWeighted(yolo_frame, 0.5, optical_flow_frame, 0.5, 0)

    return centers, yolo_frame, optical_flow_frame, combined_frame

def get_frame():
    ret, frame = cap.read()
    if not ret:
        return None
    return frame