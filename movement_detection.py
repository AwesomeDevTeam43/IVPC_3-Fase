import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Check if the video capture has been initialized correctly
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

ret, frame = cap.read()

if not ret:
    print("Error: Could not read frame.")
    cap.release()
    exit()

# setup initial location of window
x, y, w, h = 150, 200, 250, 250  # simply hardcoded the values
img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
cv2.imshow('inicial', img2)
cv2.waitKey()

track_window = (x, y, w, h)

roi = frame[y:y + h, x:x + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    if not cap.isOpened():
        cap.open(0)
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    cv2.imshow("Image Faces", frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # apply camshift to get the new location
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # Draw it on image
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv2.polylines(frame, [pts], True, 255, 2)
    print("X: " + str(track_window[0]) + " Y: " + str(track_window[1]))
    cv2.imshow('img2', img2)

    c = cv2.waitKey(1)
    if c == 27:
        break

cv2.destroyAllWindows()
cap.release()