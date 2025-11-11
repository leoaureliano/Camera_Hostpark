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
    

def insert_passagem_entrada(placa, img_path=None):
    """Registra uma nova passagem de veículo no banco"""
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO Registro_Passagens (placa_veiculo, data_hora_entrada, data_hora_saida, img_path)
        VALUES (?, ?, NULL, ?)
    """, (placa, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), img_path))
    conn.commit()
    conn.close()

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
    try:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    except:
        return None

    # upscale: aumenta resolução se a placa estiver pequena
    h, w = gray.shape
    scale = 1.0
    if h < 80:
        scale = max(2.0, 160.0 / max(1, h))
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Filtro bilateral (suaviza ruído sem perder bordas)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # CLAHE (melhora contraste local sem estourar branco/preto)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Equalização global adicional (caso tenha sombras)
    gray = cv2.equalizeHist(gray)

    # Threshold adaptativo (binarização por região)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    # Operação morfológica para unir traços quebrados
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Pequeno reforço de nitidez
    gaussian = cv2.GaussianBlur(clean, (0, 0), sigmaX=1)
    sharp = cv2.addWeighted(clean, 1.4, gaussian, -0.4, 0)

    return sharp

# ---------------- OCR wrapper (EasyOCR) ----------------
print("Inicializando EasyOCR (pode demorar na primeira vez)...")
reader = easyocr.Reader(OCR_LANGS, gpu=True)  # GPU da NVIDIA
print("EasyOCR pronto.")


def read_plate_with_easyocr(roi_bgr):
    """
    Realiza OCR com EasyOCR, aplicando múltiplas estratégias para melhorar
    leitura de placas sul-americanas.
    Retorna (texto, confiança)
    """
    proc = preprocess_roi_for_ocr(roi_bgr)
    if proc is None or proc.size == 0:
        return "", 0.0

    # Converte para RGB (EasyOCR exige isso)
    rgb = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)

    # 🔁 múltiplas tentativas de leitura com variações
    results_total = []

    # leitura base
    results = reader.readtext(rgb, detail=1, allowlist=ALLOW_CHARS)
    results_total.extend(results)

    # inversão (caso a placa esteja muito escura)
    inv = cv2.bitwise_not(rgb)
    results_inv = reader.readtext(inv, detail=1, allowlist=ALLOW_CHARS)
    results_total.extend(results_inv)

    # leve sharpen para tentar melhorar bordas de caracteres
    sharp = cv2.GaussianBlur(rgb, (0, 0), sigmaX=1)
    sharp = cv2.addWeighted(rgb, 1.5, sharp, -0.5, 0)
    results_sharp = reader.readtext(sharp, detail=1, allowlist=ALLOW_CHARS)
    results_total.extend(results_sharp)

    if not results_total:
        return "", 0.0

    # 🧠 Escolhe o texto mais confiável e consistente
    best = max(results_total, key=lambda r: r[2])
    text_raw = best[1]
    conf = float(best[2])
    text = clean_plate_text(text_raw)

    # 🚫 filtrar falsos positivos (muito curtos ou sem padrão)
    if len(text) < 6 or not plausible_plate(text):
        return "", 0.0

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

    # Verifica se há resultados válidos
    if not results or len(results) == 0:
        return None, False

    for r in results:
        for box in r.boxes:
            # caixa delimitadora
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # corte com margem
            pad_x = int(0.02 * (x2 - x1))
            pad_y = int(0.12 * (y2 - y1))
            xa = max(0, x1 - pad_x)
            ya = max(0, y1 - pad_y)
            xb = min(frame.shape[1], x2 + pad_x)
            yb = min(frame.shape[0], y2 + pad_y)
            roi = frame[ya:yb, xa:xb].copy()
            if roi.size == 0:
                continue

            # 🔍 Pré-processamento para melhorar OCR
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            gray = cv2.equalizeHist(gray)

            # OCR
            plate_text, conf = read_plate_with_easyocr(gray)
            if conf < MIN_OCR_CONF or not plausible_plate(plate_text):
                # tentativa alternativa
                plate_alt, conf_alt = read_plate_with_easyocr(roi)
                if conf_alt >= MIN_OCR_CONF and plausible_plate(plate_alt):
                    plate_text, conf = plate_alt, conf_alt
                else:
                    continue  # leitura ruim

            # ⏳ Debouncing
            now_ts = time.time()
            last = last_seen.get(plate_text)
            if last and (now_ts - last) < DEBOUNCE_SECONDS:
                if is_authorized(plate_text):
                    pass
                update_passagem_saida(plate_text)
                last_seen[plate_text] = now_ts
                continue

            # 🧠 Consultar se já existe registro aberto
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

            # 💾 Registrar passagem
            img_path = save_plate_image(roi, plate_text)
            if r_open:
                update_passagem_saida(plate_text)
                print(f"[{datetime.now()}] Saída registrada: {plate_text} (conf {conf:.2f})")
            else:
                insert_passagem_entrada(plate_text, img_path)
                print(f"[{datetime.now()}] Entrada registrada: {plate_text} (conf {conf:.2f}) img:{img_path}")

            # salvar timestamp
            last_seen[plate_text] = now_ts

            # 🎨 Desenhar no frame o retângulo e status
            autorizado = is_authorized(plate_text)
            color = (0, 255, 0) if autorizado else (0, 0, 255)
            status = "LIBERADA" if autorizado else "NEGADA"

            cv2.rectangle(frame, (xa, ya), (xb, yb), color, 2)
            cv2.putText(frame, plate_text, (xa, ya - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, status, (xa, yb + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # ✅ Se placa autorizada → mostrar popup de abertura (sem travar vídeo)
            if autorizado:
                def popup_abertura():
                    popup = tk.Toplevel()
                    popup.title("Acesso Automático - Cancela Abrindo")
                    popup.geometry("350x170")
                    popup.configure(bg="#1E1E1E")
                    popup.resizable(False, False)

                    tk.Label(popup, text=f"🚗 Placa {plate_text}",
                             bg="#1E1E1E", fg="#00FF80",
                             font=("Segoe UI", 13, "bold")).pack(pady=(15, 5))

                    tk.Label(popup, text="Cancela Abrindo...",
                             bg="#1E1E1E", fg="#00FF80",
                             font=("Segoe UI", 12)).pack(pady=(0, 10))

                    pb = ttk.Progressbar(popup, orient="horizontal",
                                         length=250, mode="determinate", maximum=100)
                    pb.pack(pady=(5, 15))

                    for i in range(101):
                        pb["value"] = i
                        popup.update_idletasks()
                        time.sleep(0.04)  # total ~4s

                    # substitui conteúdo após progresso
                    for w in popup.winfo_children():
                        w.destroy()
                    tk.Label(popup, text="✅ Acesso Liberado",
                             bg="#1E1E1E", fg="#00FF80",
                             font=("Segoe UI", 15, "bold")).pack(pady=(30, 5))
                    tk.Button(popup, text="OK", bg="#00FF80", fg="#1E1E1E",
                              font=("Segoe UI", 11, "bold"), relief="flat",
                              command=popup.destroy).pack(pady=10, ipadx=10)
                    popup.grab_set()

                threading.Thread(target=popup_abertura, daemon=True).start()

            # ✅ Retornar placa e se foi autorizada
            return plate_text, autorizado

    # 🚫 Nenhuma placa detectada
    return None, False



# ---------------- GUI + vídeo ----------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Hostpark - Controle de Acesso Veicular")
        root.geometry("750x500")
        root.configure(bg="#1E1E1E")

        # contador
        self.contador_liberadas = 0

        # widgets
        self.create_widgets()

        # inicialização da câmera
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.running = True
        self.thread = threading.Thread(target=self.loop_video, daemon=True)
        self.thread.start()

    def create_widgets(self):
        BG = "#1E1E1E"
        GREEN = "#00FF80"
        TEXT = "#FFFFFF"

        # título
        title = tk.Label(self.root, text="Hostpark - Controle de Acesso Veicular",
                         bg=BG, fg=GREEN, font=("Segoe UI", 18, "bold"))
        title.pack(pady=15)

        # container principal
        frm = tk.Frame(self.root, bg=BG)
        frm.pack(pady=10)

        # campo de cadastro
        tk.Label(frm, text="Cadastrar nova placa:", bg=BG, fg=TEXT, font=("Segoe UI", 12)).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.e = tk.Entry(frm, font=("Segoe UI", 12), width=20, justify="center")
        self.e.grid(row=0, column=1, padx=5)

        tk.Button(frm, text="Cadastrar", bg=GREEN, fg=BG, font=("Segoe UI", 12, "bold"),
                  relief="flat", command=self.on_add).grid(row=0, column=2, padx=8)

        # lista de placas
        tk.Label(frm, text="Placas cadastradas:", bg=BG, fg=TEXT, font=("Segoe UI", 12)).grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.lb = tk.Listbox(frm, bg="#2C2C2C", fg=TEXT, font=("Segoe UI", 11), width=35, height=8,
                             bd=0, selectbackground=GREEN, relief="flat")
        self.lb.grid(row=2, column=0, columnspan=2, pady=(5, 0))

        tk.Button(frm, text="Atualizar", bg=GREEN, fg=BG, font=("Segoe UI", 11),
                  relief="flat", command=self.update_list).grid(row=2, column=2, padx=8)
        tk.Button(frm, text="Parar", bg="#ff4040", fg="#fff", font=("Segoe UI", 11),
                  relief="flat", command=self.on_stop).grid(row=3, column=2, pady=(10, 0))

        self.update_list()

        # contador de placas liberadas
        contador_frame = tk.Frame(self.root, bg=BG)
        contador_frame.pack(pady=10)
        tk.Label(contador_frame, text="Placas Liberadas:", bg=BG, fg=TEXT,
                 font=("Segoe UI", 13)).pack(side="left", padx=5)
        self.contador_label = tk.Label(contador_frame, text="0", bg=BG, fg=GREEN,
                                       font=("Segoe UI", 20, "bold"))
        self.contador_label.pack(side="left")

        # botão de abertura manual
        tk.Button(self.root, text="Liberar Portão Manualmente", bg=GREEN, fg=BG,
                  font=("Segoe UI", 12, "bold"), relief="flat",
                  command=self.abrir_portao).pack(pady=15, ipadx=10, ipady=5)

        # rodapé
        tk.Label(self.root, text="© Hostpark - O Jeito Inteligente de Estacionar",
                 bg=BG, fg="#777", font=("Segoe UI", 9)).pack(side="bottom", pady=10)

    # ---------------------------------------------------
    # Atualiza lista de placas autorizadas
    def update_list(self):
        self.lb.delete(0, tk.END)
        for p in list_autorizadas():
            self.lb.insert(tk.END, p)

    # ---------------------------------------------------
    # Cadastra nova placa
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

    # ---------------------------------------------------
    # Incrementa contador (usado quando o sistema detecta placa liberada)
    def incrementar_liberadas(self):
        self.contador_liberadas += 1
        self.contador_label.config(text=str(self.contador_liberadas))

    # ---------------------------------------------------
    # Simula abertura manual do portão
    def abrir_portao(self):
        try:
            # 1) Registrar no banco como entrada manual
            insert_passagem_entrada("MANUAL", img_path="manual")

            # 2) Incrementar contador de liberações (visual)
            try:
                self.incrementar_liberadas()
            except Exception:
                if hasattr(self, "contador_label"):
                    try:
                        atual = int(self.contador_label.cget("text"))
                    except Exception:
                        atual = 0
                    self.contador_label.config(text=str(atual + 1))

            # 3) Criar popup com barra de progresso animada
            popup = tk.Toplevel(self.root)
            popup.title("Portão Aberto Manualmente")
            popup.geometry("350x170")
            popup.configure(bg="#1E1E1E")
            popup.resizable(False, False)

            tk.Label(popup, text="🚗 Cancela Abrindo...",
                     bg="#1E1E1E", fg="#00FF80",
                     font=("Segoe UI", 14, "bold")).pack(pady=(20, 10))

            pb = ttk.Progressbar(popup, orient="horizontal",
                                 length=250, mode="determinate", maximum=100)
            pb.pack(pady=(5, 15))

            def animar_barra():
                for i in range(101):
                    pb["value"] = i
                    popup.update_idletasks()
                    time.sleep(0.04)  # total ~4 segundos
                # após completar
                for w in popup.winfo_children():
                    w.destroy()
                tk.Label(popup, text="✅ Acesso Liberado",
                         bg="#1E1E1E", fg="#00FF80",
                         font=("Segoe UI", 15, "bold")).pack(pady=(30, 5))
                tk.Button(popup, text="OK", bg="#00FF80", fg="#1E1E1E",
                          font=("Segoe UI", 11, "bold"), relief="flat",
                          command=popup.destroy).pack(pady=10, ipadx=10)

            threading.Thread(target=animar_barra, daemon=True).start()
            popup.grab_set()

        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao registrar abertura manual:\n{e}")
            print("Erro registrar abertura manual:", e)
    # ---------------------------------------------------
    # Loop do vídeo + leitura de placas
    def loop_video(self):
        while self.running and not stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            try:
                plate, authorized = process_frame_and_register(frame)
                if authorized:
                    self.incrementar_liberadas()
            except Exception as e:
                print("Erro processamento frame:", e)

            # mostra janela de debug
            cv2.imshow("Hostpark - LPR", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.on_stop()
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def on_stop(self):
        self.running = False
        stop_event.set()
        self.root.quit()


# ---------------- MAIN ----------------
def main():
    ensure_dirs()
    init_db()

    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_stop)
    root.mainloop()


if __name__ == "__main__":
    main()
