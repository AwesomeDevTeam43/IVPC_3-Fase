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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                centers[label] = center_y

    if prev_gray is None:
        prev_gray = gray
        for (center_x, center_y, label) in detections:
            tracked_objects[label] = (center_x, center_y)
    else:
        if tracked_objects:
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
                    cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)
                    cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)

                    # Print the y-coordinate of the optical flow
                    print(f"Optical flow y-coordinate for {label}: {b}")
                    centers[label] = b

        for (center_x, center_y, label) in detections:
            tracked_objects[label] = (center_x, center_y)

        prev_gray = gray.copy()

    return centers

def get_frame():
    ret, frame = cap.read()
    if not ret:
        return None
    return frame