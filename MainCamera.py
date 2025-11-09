from ultralytics import YOLO
import cv2
import subprocess
import sqlite3
from datetime import datetime
import numpy as np
import os
import shutil
import time

# === Ajuste o caminho para o alpr.exe no seu PC ===
ALPR_PATH = r"C:\Program Files (x86)\openalpr-2.3.0-win-64bit\openalpr_64\alpr.exe"

# Verifica se o executável existe
if not os.path.isfile(ALPR_PATH):
    print(f"[ERRO] Executável OpenALPR não encontrado em: {ALPR_PATH}")
    print("Verifique o caminho e ajuste ALPR_PATH no topo do script.")


# === Banco de dados SQLite ===
DB_PATH = "veiculos.db"
conn = sqlite3.connect(DB_PATH, timeout=10)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS veiculos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    placa TEXT,
    cor TEXT,
    tipo_veiculo TEXT,
    data_hora_entrada TEXT
)
""")
conn.commit()

# === Função para reconhecer placa com OpenALPR ===
def reconhecer_placa(image_path):
    """
    Chama o executável alpr para reconhecer a placa na imagem imagem_path.
    Retorna a placa (string) ou None.
    """
    try:
        command = [ALPR_PATH, "-c", "br", image_path]
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, timeout=6).decode(errors="ignore")
        # debug: print(output)
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("plate0:"):
                # linha exemplo: plate0: ABC1234 confidence: 87.30
                placa = line.split(":")[1].strip().split()[0]
                return placa
    except subprocess.TimeoutExpired:
        print("[ERRO ALPR] Timeout ao executar o alpr.")
    except subprocess.CalledProcessError as e:
        print("[ERRO ALPR] CalledProcessError:", e.output.decode(errors="ignore"))
    except Exception as e:
        print("[ERRO ALPR]", e)
    return None

# === Função para detectar cor predominante ===
def detectar_cor(img):
    if img is None or img.size == 0:
        return "desconhecida"
    try:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        avg_color = np.mean(img_hsv.reshape(-1, 3), axis=0)
        h, s, v = avg_color
        if v < 50:
            return "preto"
        elif s < 60 and v > 200:
            return "branco"
        elif h < 10 or h > 160:
            return "vermelho"
        elif 10 <= h <= 25:
            return "amarelo"
        elif 25 < h <= 40:
            return "laranja"
        elif 40 < h <= 85:
            return "verde"
        elif 85 < h <= 130:
            return "azul"
        else:
            return "cinza"
    except Exception:
        return "desconhecida"

# === Função para registrar veículo no banco ===
def registrar_veiculo(placa, tipo, cor):
    data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO veiculos (placa, cor, tipo_veiculo, data_hora_entrada) VALUES (?, ?, ?, ?)",
        (placa, cor, tipo, data_hora)
    )
    conn.commit()  # commit imediato para persistência
    print(f"[INFO] Veículo {placa} ({tipo}, {cor}) regi strado às {data_hora}")

# === Função utilitária: garante crop válido e com margem ===
def safe_crop(frame, x1, y1, x2, y2, margin=10):
    h, w = frame.shape[:2]
    x1m = max(0, x1 - margin)
    y1m = max(0, y1 - margin)
    x2m = min(w - 1, x2 + margin)
    y2m = min(h - 1, y2 + margin)
    if x2m <= x1m or y2m <= y1m:
        return None
    return frame[y1m:y2m, x1m:x2m]

# === Preparar diretório temporário para imagens (limpo ao iniciar) ===
TMP_DIR = "tmp_alpr"
if os.path.isdir(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR, exist_ok=True)

# === Inicializa YOLO e câmera ===
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

line_y = 300
car_count = 0
counted_ids = set()

print("[INFO] Iniciando captura. Pressione 'q' para sair.")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERRO] Não foi possível ler frame da câmera.")
            break

        results = model.track(frame, persist=True, classes=[2, 3, 5, 7])

        if results and hasattr(results[0], "boxes") and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls in zip(boxes, ids, classes):
                name = model.names[int(cls)]
                coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], "cpu") else box.xyxy[0]
                x1, y1, x2, y2 = map(int, coords)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                placa_detectada = "?"

                # Se cruzar a linha e ainda não foi contado
                if track_id not in counted_ids and abs(cy - line_y) < 7:
                    car_count += 1
                    counted_ids.add(track_id)

                    crop = safe_crop(frame, x1, y1, x2, y2, margin=12)
                    if crop is None or crop.size == 0:
                        placa = None
                        cor = "desconhecida"
                    else:
                        tmp_name = f"{TMP_DIR}/veh_{track_id}_{int(time.time())}.jpg"
                        cv2.imwrite(tmp_name, crop)
                        placa = reconhecer_placa(tmp_name)

                        if not placa:
                            # Tenta aumentar resolução
                            big = cv2.resize(crop, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                            tmp_name2 = f"{TMP_DIR}/veh_{track_id}_big_{int(time.time())}.jpg"
                            cv2.imwrite(tmp_name2, big)
                            placa = reconhecer_placa(tmp_name2)
                            os.remove(tmp_name2)

                        os.remove(tmp_name)
                        cor = detectar_cor(crop)

                    if placa:
                        registrar_veiculo(placa, name, cor)
                        placa_detectada = placa
                        color_box = (0, 255, 0)  # verde
                    else:
                        placa_detectada = "N/A"
                        color_box = (0, 0, 255)  # vermelho

                    print(f"[INFO] {name} ID:{track_id} | Placa: {placa_detectada} | Cor: {cor}")

                # Desenha a caixa e info do veículo
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{name} ID:{track_id}", (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, f"Placa: {placa_detectada}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Linha e contador
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0,0,255), 2)
        cv2.putText(frame, f"Veículos: {car_count}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Leitor de Placas e Cores - YOLO + OpenALPR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    # opcional: limpa tmp_dir
    # shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("[INFO] Encerrado e conexão com DB fechada. DB salvo em:", os.path.abspath(DB_PATH))


