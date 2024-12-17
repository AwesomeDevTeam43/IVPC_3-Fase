import cv2
from ultralytics import YOLO
import time

# YOLO model and settings
model_path = "yolov8s.pt"
target_labels = ["cell phone", "bottle"]  # Labels que você quer detetar
model = YOLO(model_path)

# OpenCV video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

# Tracker parameters
trackers = {}  # Dicionário {label: tracker}
tracker_timeout = {}  # Tempo de inatividade para cada tracker
timeout_limit = 20  # Frames limite para resetar trackers
min_area = 1000
max_area = 100000

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y

def initialize_trackers(frame):
    """
    Reinicializa trackers usando YOLO apenas quando necessário.
    """
    global trackers, tracker_timeout

    results = model(frame, conf=0.5, verbose=False)  # Adicione 'conf' para aumentar a precisão

    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0].item()

            # Filtra por confiança e classe desejada
            if label in target_labels and confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)

                # Filtra por tamanho da área
                if min_area <= area <= max_area and label not in trackers:
                    tracker = cv2.TrackerCSRT_create()
                    trackers[label] = tracker
                    tracker_timeout[label] = 0
                    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

def update_trackers(frame):
    """
    Atualiza trackers e desenha as caixas.
    """
    global trackers, tracker_timeout

    centers = {}  # Guarda centros dos objetos
    to_remove = []

    for label, tracker in trackers.items():
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center_x, center_y = get_center((x, y, x + w, y + h))
            centers[label] = center_y

            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{label} (y={center_y})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            tracker_timeout[label] = 0  # Reseta timeout
        else:
            tracker_timeout[label] += 1
            if tracker_timeout[label] > timeout_limit:
                to_remove.append(label)  # Remove trackers fantasmas

    # Remove trackers que perderam os objetos
    for label in to_remove:
        trackers.pop(label)
        tracker_timeout.pop(label)

    return centers

# Main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inicializa ou reinicializa trackers periodicamente
        if len(trackers) == 0 or any(v > timeout_limit for v in tracker_timeout.values()):
            initialize_trackers(frame)

        centers = update_trackers(frame)  # Atualiza os trackers

        # Mostra FPS
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Exibe o vídeo
        cv2.imshow("Tracking com YOLO e OpenCV", frame)

        # Pressione "q" para sair
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
