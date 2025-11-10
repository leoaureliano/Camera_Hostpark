"""
lpr_yolo_easyocr.py
Sistema de Controle de Acesso Veicular:
- Detecção de placa com YOLO (Ultralytics)
- OCR com EasyOCR
- Tracker simples por centróide para manter identidade entre frames
- Banco de dados SQLite com tabelas Carros_Autorizados e Registro_Passagens
- GUI Tkinter para cadastro/listagem e abertura manual de cancela
- Salva imagens das placas detectadas em ./plates/

Recomendações: testar em ambiente bem iluminado e câmera fixa.
"""

import os
import cv2
import time
import math
import sqlite3
import threading
from datetime import datetime
import numpy as np
import easyocr
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image
from ultralytics import YOLO

# Caminho do modelo — use o yolov8n.pt baixado
MODEL_NAME_OR_PATH = "yolov8n.pt"

# Carrega o modelo uma única vez no início do script
yolo_model = YOLO(MODEL_NAME_OR_PATH)
print("✅ Modelo YOLO carregado com sucesso!")

# -------------------- CONFIGURAÇÃO --------------------
CAM_INDEX = 0  # ou caminho para vídeo
COUNTING_LINE_Y = 360  # posição da linha de contagem (ajuste conforme resolução)
DB_FILENAME = "controle_acesso.db"
PLATES_DIR = "plates"
MODEL_NAME_OR_PATH = "yolov8n.pt"
MODEL_NAME_OR_PATH = "https://github.com/ultralytics/assets/releases/download/v8.0.0/license_plate_detector.pt"
OCR_LANGS = ['en']  # normalmente 'en' funciona bem para placas; também pode usar ['pt'] se quiser

# Parâmetros do tracker
MAX_LOST_FRAMES = 10  # depois de quantos frames sem detecção um objeto é descartado
DIST_THRESHOLD = 120  # distância máxima (px) para associar detecções ao tracker

# OCR config (cleaning)
import re
_alnum_re = re.compile(r'[^A-Z0-9]')

# Locks
db_lock = threading.Lock()
count_lock = threading.Lock()
stop_event = threading.Event()

# -------------------- UTILIDADES --------------------
def ensure_dirs():
    if not os.path.exists(PLATES_DIR):
        os.makedirs(PLATES_DIR)

def clean_plate_text(text):
    if not text:
        return ""
    s = text.upper()
    s = _alnum_re.sub('', s)
    # correções simples comuns
    s = s.replace('O', '0')
    s = s.replace('I', '1')
    s = s.replace(' ', '')
    if len(s) > 10:
        s = s[:10]
    return s

def plausible_plate(s):
    if not s:
        return False
    if not (4 <= len(s) <= 10):
        return False
    has_digit = any(c.isdigit() for c in s)
    has_alpha = any(c.isalpha() for c in s)
    return has_digit and has_alpha

# -------------------- BANCO DE DADOS --------------------
def init_db():
    with db_lock:
        conn = sqlite3.connect(DB_FILENAME, check_same_thread=False)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Carros_Autorizados (
                placa_veiculo TEXT PRIMARY KEY
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Registro_Passagens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                placa_veiculo TEXT,
                data_hora_entrada TEXT,
                data_hora_saida TEXT,
                contador_total INTEGER,
                img_path TEXT
            )
        """)
        conn.commit()
        # inserir alguns dados de teste (ignora duplicatas)
        try:
            cur.executemany("INSERT INTO Carros_Autorizados (placa_veiculo) VALUES (?)",
                            [("ABC1234",), ("XYZ9876",), ("CAR2025",)])
        except sqlite3.IntegrityError:
            pass
        conn.commit()
        conn.close()

def add_autorizada(placa):
    placa = placa.upper()
    with db_lock:
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO Carros_Autorizados (placa_veiculo) VALUES (?)", (placa,))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

def list_autorizadas():
    with db_lock:
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute("SELECT placa_veiculo FROM Carros_Autorizados ORDER BY placa_veiculo")
        rows = [r[0] for r in cur.fetchall()]
        conn.close()
        return rows

def insert_passagem(placa, entrada_time, img_path=None):
    with db_lock:
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM Registro_Passagens")
        cnt = cur.fetchone()[0] + 1
        cur.execute("""
            INSERT INTO Registro_Passagens (placa_veiculo, data_hora_entrada, data_hora_saida, contador_total, img_path)
            VALUES (?, ?, ?, ?, ?)
        """, (placa, entrada_time, None, cnt, img_path))
        conn.commit()
        conn.close()

def update_saida(placa, saida_time):
    with db_lock:
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute("""
            SELECT id FROM Registro_Passagens
            WHERE placa_veiculo = ? AND data_hora_saida IS NULL
            ORDER BY id DESC LIMIT 1
        """, (placa,))
        r = cur.fetchone()
        if r:
            rec_id = r[0]
            cur.execute("UPDATE Registro_Passagens SET data_hora_saida = ? WHERE id = ?", (saida_time, rec_id))
            conn.commit()
        conn.close()

def is_authorized(placa):
    with db_lock:
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM Carros_Autorizados WHERE placa_veiculo = ?", (placa,))
        r = cur.fetchone()
        conn.close()
        return r is not None

# -------------------- TRACKER SIMPLES (CENTROID TRACKER) --------------------
class TrackedObject:
    def __init__(self, obj_id, bbox, centroid, last_seen_frame):
        self.id = obj_id
        self.bbox = bbox  # (x,y,w,h)
        self.centroid = centroid
        self.last_seen = last_seen_frame
        self.lost_frames = 0
        self.plate_text = None
        self.counted = False

class SimpleTracker:
    def __init__(self, dist_threshold=DIST_THRESHOLD, max_lost=MAX_LOST_FRAMES):
        self.next_object_id = 1
        self.objects = dict()  # id -> TrackedObject
        self.dist_threshold = dist_threshold
        self.max_lost = max_lost

    def update(self, detections, frame_idx):
        """
        detections: list of tuples (x,y,w,h,score)
        returns list of TrackedObject currently active
        """
        centroids = []
        for (x,y,w,h,score) in detections:
            cx = x + w//2
            cy = y + h//2
            centroids.append((cx,cy,(x,y,w,h)))

        if len(self.objects) == 0:
            for c in centroids:
                self.objects[self.next_object_id] = TrackedObject(self.next_object_id, c[2], (c[0], c[1]), frame_idx)
                self.next_object_id += 1
        else:
            # match by minimal distance
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[oid].centroid for oid in object_ids]
            used_det = set()
            # compute distance matrix
            for oid, (ox,oy) in zip(object_ids, object_centroids):
                best_det = None
                best_dist = None
                for i, (cx,cy,bbox) in enumerate(centroids):
                    if i in used_det: 
                        continue
                    d = math.hypot(ox - cx, oy - cy)
                    if best_dist is None or d < best_dist:
                        best_dist = d
                        best_det = i
                if best_det is not None and best_dist <= self.dist_threshold:
                    # update object
                    cx, cy, bbox = centroids[best_det]
                    tobj = self.objects[oid]
                    tobj.centroid = (cx,cy)
                    tobj.bbox = bbox
                    tobj.last_seen = frame_idx
                    tobj.lost_frames = 0
                    used_det.add(best_det)
                else:
                    # not matched -> increment lost
                    self.objects[oid].lost_frames += 1

            # create new objects for unmatched detections
            for i, (cx,cy,bbox) in enumerate(centroids):
                if i not in used_det:
                    self.objects[self.next_object_id] = TrackedObject(self.next_object_id, bbox, (cx,cy), frame_idx)
                    self.next_object_id += 1

        # remove objects that exceeded lost_frames
        to_remove = [oid for oid, o in self.objects.items() if o.lost_frames > self.max_lost]
        for oid in to_remove:
            del self.objects[oid]

        return list(self.objects.values())

# -------------------- INICIALIZAÇÃO MODELOS --------------------
print("Inicializando modelos... (pode demorar um pouco na 1a execução)")
reader = easyocr.Reader(OCR_LANGS, gpu=False)  # gpu=True se tiver suporte
print("Modelos prontos.")

# -------------------- PROCESSAMENTO DE ROI -> OCR --------------------
def preprocess_roi_for_ocr(roi):
    # roi: BGR numpy
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # aumentar resolução
    h, w = gray.shape[:2]
    if h < 80:
        scale = max(2, int(160 / h))
        gray = cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    # equalize
    gray = cv2.equalizeHist(gray)
    # denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # threshold
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def ocr_easyocr(roi):
    # roi: color BGR
    img_proc = preprocess_roi_for_ocr(roi)
    # easyocr espera imagem RGB ou caminho; converte para RGB
    img_rgb = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2RGB)
    results = reader.readtext(img_rgb, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    if results:
        # escolher o resultado mais longo/mais plausível
        results = [clean_plate_text(r) for r in results]
        results = [r for r in results if r]
        if not results:
            return ""
        # preferência para as que parecem placa
        results_sorted = sorted(results, key=lambda r: (plausible_plate(r), len(r)), reverse=True)
        return results_sorted[0]
    return ""

# -------------------- LÓGICA DE CONTAGEM E REGISTRO --------------------
global_count = 0
tracker = SimpleTracker()

def save_plate_image(plate_img, plate_text):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_text = plate_text if plate_text else "UNKNOWN"
    filename = f"{safe_text}_{ts}.png"
    path = os.path.join(PLATES_DIR, filename)
    cv2.imwrite(path, plate_img)
    return path

def handle_tracked_objects(tracked_objs, frame, frame_idx):
    global global_count
    h_frame = frame.shape[0]
    for tobj in tracked_objs:
        x,y,w,h = tobj.bbox
        cx,cy = tobj.centroid
        # se ainda não leu placa para esse objeto, faça OCR
        if not tobj.plate_text:
            try:
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                text = ocr_easyocr(roi)
                if plausible_plate(text):
                    tobj.plate_text = text
                    # salvar imagem
                    img_path = save_plate_image(roi, text)
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Detected plate: {text} saved:{img_path}")
                else:
                    # opcional: guardar tentativa
                    tobj.plate_text = None
            except Exception as e:
                print("Erro OCR:", e)
        # contagem por travessia da linha
        if tobj.plate_text:
            # detecta travessia apenas uma vez por passagem
            if not tobj.counted:
                # se o centro cruzou a linha (simples heurística): centro y < line e agora > line OR vice-versa
                # para isso guardamos last_seen (frame index) e para simplificar contamos quando cy cruza linha
                if cy >= COUNTING_LINE_Y - 5 and cy <= COUNTING_LINE_Y + 5:
                    # atravessou a linha; decidir entrada/saida: look up active registros
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # se há registro aberto -> saída, se não -> entrada
                    with db_lock:
                        conn = sqlite3.connect(DB_FILENAME)
                        cur = conn.cursor()
                        cur.execute("""
                            SELECT id FROM Registro_Passagens
                            WHERE placa_veiculo = ? AND data_hora_saida IS NULL
                            ORDER BY id DESC LIMIT 1
                        """, (tobj.plate_text,))
                        r = cur.fetchone()
                        conn.close()
                    if r is None:
                        # entrada
                        img_path = save_plate_image(frame[y:y+h, x:x+w], tobj.plate_text)
                        insert_passagem(tobj.plate_text, now, img_path)
                        with count_lock:
                            global_count += 1
                        print(f"[{now}] Entrada registrada para {tobj.plate_text}")
                    else:
                        # saída
                        update_saida(tobj.plate_text, now)
                        print(f"[{now}] Saída registrada para {tobj.plate_text}")
                    tobj.counted = True

# -------------------- VÍDEO LOOP --------------------
def video_loop(app_ref):
    ensure_dirs()
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        messagebox.showerror("Erro", f"Não foi possível abrir a câmera (CAM_INDEX={CAM_INDEX}).")
        return

    # tenta configurar resolução (opcional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_idx = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        # rodar detector YOLO cada N frames para performance (ex: 1 em 2)
        results = yolo_model.predict(frame, imgsz=640, conf=0.35, verbose=False, device='cpu')
        detections = []
        # cada result contem boxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
        # classe 2 = carro, 3 = moto, 5 = ônibus, 7 = caminhão
                if cls in [2, 3, 5, 7] and conf > 0.4:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    detections.append((x1, y1, w, h, conf))

        # atualizar tracker com detections
        tracked = tracker.update(detections, frame_idx)

        # desenhar e processar cada objeto trackeado
        for tobj in tracked:
            x,y,w,h = tobj.bbox
            cx,cy = tobj.centroid
            # aparência no frame
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            label = f"ID{tobj.id}"
            if tobj.plate_text:
                label += f" {tobj.plate_text}"
                # indicar autorização
                if is_authorized(tobj.plate_text):
                    label += " (AUT)"
                    cv2.putText(frame, "AUTORIZADO", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                else:
                    cv2.putText(frame, "NAO AUTORIZADO", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # executar manipulações baseadas nos tracked objects (OCR + contagem)
        handle_tracked_objects(tracked, frame, frame_idx)

        # desenhar linha de contagem e contador
        cv2.line(frame, (0, COUNTING_LINE_Y), (frame.shape[1], COUNTING_LINE_Y), (255,0,0), 2)
        with count_lock:
            ctext = f"Total passagens: {global_count}"
        cv2.putText(frame, ctext, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        # exibir número de objetos rastreados
        cv2.putText(frame, f"Tracked: {len(tracked)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.imshow("LPR - YOLO + EasyOCR (press q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break

    cap.release()
    #cv2.destroyAllWindows()

# -------------------- GUI --------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Controle de Acesso Veicular - YOLO + EasyOCR")
        root.geometry("760x480")
        self.create_widgets()
        self.video_thread = threading.Thread(target=video_loop, args=(self,), daemon=True)
        self.video_thread.start()
        self.update_autorizadas_list_periodic()

    def create_widgets(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # cadastro
        ttk.Label(frm, text="Cadastrar placa:").grid(column=0, row=0, sticky=tk.W)
        self.entry_placa = ttk.Entry(frm, width=20)
        self.entry_placa.grid(column=1, row=0, sticky=tk.W)
        ttk.Button(frm, text="Adicionar", command=self.on_add).grid(column=2, row=0, sticky=tk.W)

        # listagem
        ttk.Label(frm, text="Placas autorizadas:").grid(column=0, row=1, sticky=tk.W, pady=(10,0))
        self.listbox = tk.Listbox(frm, height=12, width=25)
        self.listbox.grid(column=0, row=2, rowspan=6, sticky=tk.W)

        # abertura manual
        ttk.Label(frm, text="Abertura manual (placa):").grid(column=1, row=2, sticky=tk.W)
        self.entry_manual = ttk.Entry(frm, width=20)
        self.entry_manual.grid(column=1, row=3, sticky=tk.W)
        ttk.Button(frm, text="Abrir Cancela", command=self.on_manual_open).grid(column=2, row=3, sticky=tk.W)

        ttk.Button(frm, text="Atualizar lista", command=self.update_autorizadas_list).grid(column=1, row=4, sticky=tk.W, pady=(8,0))
        ttk.Button(frm, text="Mostrar últimos registros (console)", command=self.print_registros).grid(column=1, row=5, sticky=tk.W, pady=(8,0))
        ttk.Button(frm, text="Parar sistema", command=self.on_stop).grid(column=2, row=5, sticky=tk.W, pady=(8,0))

    def on_add(self):
        placa = self.entry_placa.get().strip().upper()
        placa = clean_plate_text(placa)
        if not placa:
            messagebox.showwarning("Aviso", "Placa inválida.")
            return
        ok = add_autorizada(placa)
        if ok:
            messagebox.showinfo("Sucesso", f"Placa {placa} adicionada.")
            self.entry_placa.delete(0, tk.END)
            self.update_autorizadas_list()
        else:
            messagebox.showwarning("Aviso", f"Placa {placa} já existe.")

    def update_autorizadas_list(self):
        placas = list_autorizadas()
        self.listbox.delete(0, tk.END)
        for p in placas:
            self.listbox.insert(tk.END, p)

    def update_autorizadas_list_periodic(self):
        self.update_autorizadas_list()
        if not stop_event.is_set():
            self.root.after(3000, self.update_autorizadas_list_periodic)

    def on_manual_open(self):
        placa = self.entry_manual.get().strip().upper()
        placa = clean_plate_text(placa)
        if not placa:
            messagebox.showwarning("Aviso", "Placa inválida.")
            return
        if is_authorized(placa):
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insert_passagem(placa, now, None)
            global global_count
            with count_lock:
                global_count += 1
            messagebox.showinfo("Cancela", f"Cancela aberta manualmente para {placa}.")
        else:
            messagebox.showwarning("Cancela", f"Placa {placa} não autorizada.")

    def print_registros(self):
        with db_lock:
            conn = sqlite3.connect(DB_FILENAME)
            cur = conn.cursor()
            cur.execute("SELECT id, placa_veiculo, data_hora_entrada, data_hora_saida, contador_total, img_path FROM Registro_Passagens ORDER BY id DESC LIMIT 50")
            rows = cur.fetchall()
            conn.close()
        print("===== Últimos registros =====")
        for r in rows:
            print(r)
        messagebox.showinfo("Registros", "Últimos registros exibidos no console.")

    def on_stop(self):
        if messagebox.askyesno("Confirmar", "Parar o sistema?"):
            stop_event.set()
            self.root.after(500, self.root.quit)

# -------------------- MAIN --------------------
def main():
    ensure_dirs()
    init_db()
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (stop_event.set(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()
