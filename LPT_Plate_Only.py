"""
lpr_plate_only.py
Detecta SOMENTE placas (modelo LPR), faz OCR focado (A-Z,0-9),
e registra no SQLite (entrada/saída). Interface Tkinter mínima.
Tem tratamento para modelo corrompido e pré-processamento robusto.
"""

import os
import cv2
import time
import sqlite3
import threading
import requests
from datetime import datetime
import numpy as np
import easyocr
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox
import re

# ---------------- CONFIG ----------------
MODEL_LOCAL = "license_plate_detector.pt"
MODEL_URL = "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8?utm_source=chatgpt.com"
CAM_INDEX = 0
DB_FILENAME = "controle_acesso.db"
PLATES_DIR = "plates"
OCR_LANGS = ['pt', 'en']   # tenta pt + en (bom para Sulamérica)
ALLOW_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MIN_OCR_CONF = 0.45       # confiança mínima do OCR para aceitar leitura
MIN_PLATE_LEN = 4
MAX_PLATE_LEN = 10
# ---------------- end config ----------------

# locks / flags
stop_event = threading.Event()
db_lock = threading.Lock()
count_lock = threading.Lock()

# regex e limpeza
_alnum_re = re.compile(r'[^A-Z0-9]')

def ensure_dirs():
    os.makedirs(PLATES_DIR, exist_ok=True)

def clean_plate_text(text):
    if not text: return ""
    s = text.upper()
    s = _alnum_re.sub('', s)
    # correções simples
    s = s.replace('O', '0').replace('I', '1')
    if len(s) > MAX_PLATE_LEN: s = s[:MAX_PLATE_LEN]
    return s

def plausible_plate(s):
    if not s: return False
    if not (MIN_PLATE_LEN <= len(s) <= MAX_PLATE_LEN): return False
    has_digit = any(c.isdigit() for c in s)
    has_alpha = any(c.isalpha() for c in s)
    return has_digit and has_alpha

# ---------------- SQLite ----------------
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
                img_path TEXT
            )
        """)
        conn.commit()
        # dados de exemplo
        try:
            cur.executemany("INSERT INTO Carros_Autorizados (placa_veiculo) VALUES (?)",
                            [("ABC1234",), ("XYZ9876",)])
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

def insert_passagem_entrada(placa, img_path=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with db_lock:
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO Registro_Passagens (placa_veiculo, data_hora_entrada, data_hora_saida, img_path)
            VALUES (?, ?, NULL, ?)
        """, (placa, now, img_path))
        conn.commit()
        conn.close()

def update_passagem_saida(placa):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
            cur.execute("UPDATE Registro_Passagens SET data_hora_saida = ? WHERE id = ?", (now, r[0]))
            conn.commit()
        conn.close()

def is_authorized(placa):
    with db_lock:
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM Carros_Autorizados WHERE placa_veiculo = ?", (placa,))
        ok = cur.fetchone()
        conn.close()
        return ok is not None

# ---------------- Model utils ----------------
def download_model(url=MODEL_URL, dest=MODEL_LOCAL, chunk_size=8192):
    print("Baixando modelo LPR (pode demorar)...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    print("Download concluído:", dest)

def load_yolo_model(path=MODEL_LOCAL):
    # tenta carregar; se falhar por arquivo corrompido, remove e baixa
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        print("Erro ao carregar modelo:", e)
        if os.path.exists(path):
            try:
                os.remove(path)
                print("Arquivo de modelo corrompido removido:", path)
            except:
                pass
        # tenta baixar e recarregar
        download_model()
        model = YOLO(path)
        return model

# ---------------- OCR preprocessing ----------------
def preprocess_roi_for_ocr(roi_bgr):
    # roi_bgr: color BGR crop da placa
    try:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    except:
        return None
    # upscale para melhorar leitura
    h, w = gray.shape
    scale = 1.0
    if h < 80:
        scale = max(2.0, 160.0 / max(1, h))
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # denoise e equalize
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.equalizeHist(gray)
    # sharpen (unsharp mask)
    gaussian = cv2.GaussianBlur(gray, (0,0), sigmaX=3)
    sharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    # adaptive threshold for variable lighting
    th = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 41, 15)
    # morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    return clean

# ---------------- OCR wrapper (EasyOCR) ----------------
print("Inicializando EasyOCR (pode demorar na primeira vez)...")
reader = easyocr.Reader(OCR_LANGS, gpu=False)  # troca para gpu=True se tiver CUDA
print("EasyOCR pronto.")

def read_plate_with_easyocr(roi_bgr):
    proc = preprocess_roi_for_ocr(roi_bgr)
    if proc is None:
        return "", 0.0
    # EasyOCR aceita RGB
    rgb = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
    # detail=1 para obter confiança
    results = reader.readtext(rgb, detail=1, allowlist=ALLOW_CHARS)
    if not results:
        return "", 0.0
    # escolher resultado com maior confiança * e limpar
    best = max(results, key=lambda r: r[2])
    text_raw = best[1]
    conf = float(best[2])
    text = clean_plate_text(text_raw)
    return text, conf

# ---------------- core: captura e lógica ----------------
ensure_dirs()
init_db()

print("Carregando modelo YOLO (LPR)...")
if not os.path.exists(MODEL_LOCAL):
    try:
        download_model()
    except Exception as e:
        print("Falha no download automático do modelo. Por favor baixe manualmente:", MODEL_URL)
        raise

yolo_model = load_yolo_model(MODEL_LOCAL)
print("Modelo LPR pronto.")

# minimal tracker: guarda última placa vista e tempo para evitar múltiplos registros rápidos
last_seen = {}  # placa -> timestamp_of_last_detection

DEBOUNCE_SECONDS = 5  # não registrar a mesma placa mais de uma vez dentro disso (ajuste)

def save_plate_image(roi_bgr, plate_text):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = plate_text if plate_text else "UNKNOWN"
    fname = f"{safe}_{ts}.png"
    path = os.path.join(PLATES_DIR, fname)
    cv2.imwrite(path, roi_bgr)
    return path

def process_frame_and_register(frame):
    global last_seen
    results = yolo_model.predict(frame, imgsz=640, conf=0.35, verbose=False)
    # resultados provavelmente contém boxes de placas (um modelo LPR)
    for r in results:
        for box in r.boxes:
            # caixa
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            # corte com margem
            pad_x = int(0.02 * (x2-x1))
            pad_y = int(0.12 * (y2-y1))
            xa = max(0, x1 - pad_x); ya = max(0, y1 - pad_y)
            xb = min(frame.shape[1], x2 + pad_x); yb = min(frame.shape[0], y2 + pad_y)
            roi = frame[ya:yb, xa:xb].copy()
            if roi.size == 0:
                continue

            # OCR
            plate_text, conf = read_plate_with_easyocr(roi)
            if conf < MIN_OCR_CONF or not plausible_plate(plate_text):
                # tentativa alternativa: tentar OCR no ROI colorido sem threshold (às vezes ajuda)
                plate_alt, conf_alt = read_plate_with_easyocr(roi)  # já preprocessa; mantemos tentativa só uma vez
                if conf_alt >= MIN_OCR_CONF and plausible_plate(plate_alt):
                    plate_text, conf = plate_alt, conf_alt
                else:
                    # não confiável -> pular
                    continue

            # debouncing: evita registrar muitas vezes seguidas
            now_ts = time.time()
            last = last_seen.get(plate_text)
            if last and (now_ts - last) < DEBOUNCE_SECONDS:
                # já registrado recentemente -> atualiza saída se existir registro aberto
                if is_authorized(plate_text):
                    # opcional: apenas notificar
                    pass
                # tenta fechar registro (se houver)
                update_passagem_saida(plate_text)
                last_seen[plate_text] = now_ts
                continue

            # salvar imagem da placa
            img_path = save_plate_image(roi, plate_text)
            # registrar entrada ou saída:
            # se já tem registro aberto (sem saída) -> marcar saída; senão inserir entrada
            with db_lock:
                conn = sqlite3.connect(DB_FILENAME)
                cur = conn.cursor()
                cur.execute("""
                    SELECT id FROM Registro_Passagens
                    WHERE placa_veiculo = ? AND data_hora_saida IS NULL
                    ORDER BY id DESC LIMIT 1
                """, (plate_text,))
                r_open = cur.fetchone()
                conn.close()
            if r_open:
                update_passagem_saida(plate_text)
                print(f"[{datetime.now()}] Saída registrada: {plate_text} (conf {conf:.2f})")
            else:
                insert_passagem_entrada(plate_text, img_path)
                print(f"[{datetime.now()}] Entrada registrada: {plate_text} (conf {conf:.2f}) img:{img_path}")

            # salvar timestamp
            last_seen[plate_text] = now_ts

            # desenhar no frame para debug e status de acesso
            color = (0, 255, 0) if is_authorized(plate_text) else (0, 0, 255)
            status = "LIBERADA" if is_authorized(plate_text) else "NEGADA"

            cv2.rectangle(frame, (xa, ya), (xb, yb), color, 2)
            cv2.putText(frame, plate_text, (xa, ya - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, status, (xa, yb + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# ---------------- GUI + vídeo ----------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("LPR - Placas Apenas")
        self.create_widgets()
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.running = True
        self.thread = threading.Thread(target=self.loop_video, daemon=True)
        self.thread.start()

    def create_widgets(self):
        frm = ttk.Frame(self.root, padding=8)
        frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="Cadastrar placa autorizada:").grid(row=0, column=0, sticky=tk.W)
        self.e = ttk.Entry(frm, width=20); self.e.grid(row=0, column=1)
        ttk.Button(frm, text="Adicionar", command=self.on_add).grid(row=0, column=2)
        ttk.Label(frm, text="Placas cadastradas:").grid(row=1, column=0, sticky=tk.W, pady=(8,0))
        self.lb = tk.Listbox(frm, height=8, width=30); self.lb.grid(row=2, column=0, columnspan=2)
        ttk.Button(frm, text="Atualizar", command=self.update_list).grid(row=2, column=2)
        ttk.Button(frm, text="Parar", command=self.on_stop).grid(row=3, column=2, pady=(8,0))
        self.update_list()

    def update_list(self):
        self.lb.delete(0, tk.END)
        for p in list_autorizadas():
            self.lb.insert(tk.END, p)

    def on_add(self):
        v = self.e.get().strip().upper()
        v = clean_plate_text(v)
        if not plausible_plate(v):
            messagebox.showwarning("Aviso", "Placa inválida.")
            return
        ok = add_autorizada(v)
        if ok:
            messagebox.showinfo("Sucesso", f"Placa {v} adicionada.")
            self.update_list()
        else:
            messagebox.showwarning("Aviso", "Placa já existe.")

    def loop_video(self):
        while self.running and not stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            # processa frame e registra placas confiáveis
            try:
                process_frame_and_register(frame)
            except Exception as e:
                print("Erro processamento frame:", e)
            # mostra (janela OpenCV) para debug
            cv2.imshow("LPR - Placas Apenas (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.on_stop()
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def on_stop(self):
        self.running = False
        stop_event.set()
        self.root.quit()

def main():
    ensure_dirs()
    init_db()
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_stop)
    root.mainloop()

if __name__ == "__main__":
    main()
