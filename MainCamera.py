from ultralytics import YOLO
import cv2

# Carrega o modelo YOLO pré-treinado (use 'yolov8n.pt' para começar)
model = YOLO("yolov8n.pt")

# Abre a câmera (0 = webcam local)
cap = cv2.VideoCapture(0)

# Coordenadas da linha de contagem
line_y = 300
car_count = 0

# Armazena IDs já contados
counted_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecta e rastreia objetos
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7])  
    # Classes: 2=car, 3=motorcycle, 5=bus, 7=truck (COCO dataset)

    # Acessa as detecções
    if results[0].boxes.id is not None:
        boxes = results[0].boxes
        ids = boxes.id.cpu().numpy().astype(int)
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, classes):
            name = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Se o veículo cruzar a linha e ainda não foi contado
            if track_id not in counted_ids and abs(cy - line_y) < 5:
                car_count += 1
                counted_ids.add(track_id)

            # Desenha bounding box e ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{name} ID:{track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Desenha linha e contador
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0,0,255), 2)
    cv2.putText(frame, f"Carros: {car_count}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Contador de Carros", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
