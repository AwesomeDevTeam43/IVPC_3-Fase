import cv2
from ultralytics import YOLO
import numpy as np

# defenir parametros para o YOLO
model_path = "yolov8l.pt"
target_labels = ["cell phone", "bottle"]
model = YOLO(model_path)

# Abrir camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():  # verifica se a camera abriu
    print("Error accessing the camera.")
    exit()

# Definir a resolução da camera
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Parametros para o algoritmo Lucas-Kanade Optical Flow
lk_params = dict(
    winSize=(15, 15),  # Tamanho da janela de análise
    maxLevel=2,  # Níveis na piramide de imagens
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Critérios -> precisão | número de iterações
)

prev_gray = None  # variavel para guardar o frame anterior
tracked_objects = {}  # variavel para guardar posição dos objetos


# Função para calcular o ponto central de uma bounding box
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y


# Função principal para detetar e rastrear objetos
def detect_and_track(frame):
    global prev_gray, tracked_objects

    centers = {}  # Variavel para armazenar as coordenadas y dos objetos

    # criar frames para exibir resultados
    yolo_frame = frame.copy()
    optical_flow_frame = frame.copy()
    combined_frame = frame.copy()

    # converçao do frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # deteção de objetos com YOLO
    results = model(frame, conf=0.7, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0].item()
            if label in target_labels and confidence > 0.7:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x, center_y = get_center((x1, y1, x2, y2))
                detections.append((center_x, center_y, label))
                # mostrar resultados no frame
                cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(yolo_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                centers[label] = center_y
                cv2.putText(yolo_frame, f"y: {center_y}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # verifica se o frame anterior existe, senao guardar o frame atual
    if prev_gray is None:
        prev_gray = gray
        for (center_x, center_y, label) in detections:
            tracked_objects[label] = (center_x, center_y)
    #
    else:
        # Calcular o Optical Flow para objetos rastreados
        if tracked_objects:
            y_offset = 30
            for label, point in tracked_objects.items():
                # Converter o ponto atual para o formato ideal pra o Optical Flow
                prev_points = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
                good_new = next_points[status == 1]  #Novos pontos rastreados
                good_old = prev_points[status == 1]  #Pontos anteriores

                if len(good_new) > 0:
                    #Atualiza as posiçoes dos pontos rastreados
                    new = good_new[0]
                    old = good_old[0]
                    a, b = new.ravel()  #Nova posição
                    c, d = old.ravel()  #Posição anterior
                    a, b = int(a), int(b)
                    c, d = int(c), int(d)
                    tracked_objects[label] = (a, b)

                    # mostrar resultados no frame
                    cv2.circle(optical_flow_frame, (a, b), 5, (0, 255, 0), -1)
                    cv2.line(optical_flow_frame, (a, b), (c, d), (0, 255, 0), 2)

                    # mostrar a coordenada y do ponto rastreado no console
                    print(f"Optical flow y-coordinate for {label}: {b}")
                    centers[label] = b
                    # mostrar a coordenada y no frame do Optical Flow
                    cv2.putText(optical_flow_frame, f"y: {b}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                2)
                    y_offset += 30

        # adicionar novas deteções ao rastreamento
        for (center_x, center_y, label) in detections:
            tracked_objects[label] = (center_x, center_y)

        # atualiza o frame anterior
        prev_gray = gray.copy()

    # Combinar os resultados do YOLO e do Optical Flow
    combined_frame = cv2.addWeighted(yolo_frame, 0.5, optical_flow_frame, 0.5, 0)

    return centers, yolo_frame, optical_flow_frame, combined_frame



def get_frame():
    ret, frame = cap.read()  # Ler um frame da câmara
    if not ret:
        return None
    return frame
