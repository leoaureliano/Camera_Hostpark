"""
lpr_control_acesso.py
Protótipo de Controle de Acesso Veicular com LPR básico, SQLite e Tkinter.
"""

import cv2
import numpy as np
import pytesseract
import sqlite3
import threading
import time
from datetime import datetime
import re
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

#chama o tesseract:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Parâmetros do sistema:
CAM_INDEX = 0  # índice da webcam
COUNTING_LINE_Y = 300  # posição da linha de contagem (px)
MIN_PLATE_WIDTH = 60
MIN_PLATE_HEIGHT = 15

DB_FILENAME = "controle_acesso.db"

# OCR config (tenta reduzir falsos positivos)
TESS_CONFIG = r"-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7"

# -------------------- BANCO DE DADOS --------------------
db_lock = threading.Lock()

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
                contador_total INTEGER
            )
        """)
        conn.commit()
        # Insere alguns dados de teste
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

def insert_passagem(placa, entrada_time):
    with db_lock:
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        # contador_total = número total de registros (após inserção)
        cur.execute("SELECT COUNT(*) FROM Registro_Passagens")
        cnt = cur.fetchone()[0] + 1
        cur.execute("""
            INSERT INTO Registro_Passagens (placa_veiculo, data_hora_entrada, data_hora_saida, contador_total)
            VALUES (?, ?, ?, ?)
        """, (placa, entrada_time, None, cnt))
        conn.commit()
        conn.close()

def update_saida(placa, saida_time):
    with db_lock:
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        # encontra último registro sem saída para a placa
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

# -------------------- UTILIDADES OCR / PLACA --------------------
alnum_re = re.compile(r'[^A-Z0-9]')

def clean_plate_text(text):
    if not text:
        return ""
    s = text.upper()
    s = alnum_re.sub('', s)
    # tentativa simples de correção (ex.: O -> 0, I -> 1) - opcional
    s = s.replace('O', '0')
    s = s.replace('I', '1')
    # limitar comprimento razoável (placas BR geralmente 7)
    if len(s) > 9:
        s = s[:9]
    return s

def plausible_plate(s):
    # Critério simples: comprimento entre 5 e 8 e pelo menos 1 letra e 1 dígito
    if not s:
        return False
    if not (5 <= len(s) <= 9):
        return False
    has_digit = any(c.isdigit() for c in s)
    has_alpha = any(c.isalpha() for c in s)
    return has_digit and has_alpha

def ocr_plate_from_roi(roi):
    # roi é imagem colorida no formato BGR (OpenCV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # técnicas de pré-processamento
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # aumenta contraste caso seja necessário (opcional)
    # usar PIL / pytesseract direto
    text = pytesseract.image_to_string(th, config=TESS_CONFIG)
    text = clean_plate_text(text)
    return text

# -------------------- DETECÇÃO SIMPLES DE PLACA (heurística) --------------------
def detect_plate_candidates(frame):
    """
    Retorna lista de bounding boxes prováveis de placa (x,y,w,h).
    Método heurístico: edge detection e busca por retângulos com aspecto de placa.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # equalização / blur
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.06 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if w >= MIN_PLATE_WIDTH and h >= MIN_PLATE_HEIGHT:
                aspect = w / float(h)
                # aspecto típico placa ~ 2 to 5 (ajustar conforme padrão)
                if 2.0 <= aspect <= 6.0 and w > 80:
                    candidates.append((x, y, w, h))
    # ordenar por área decrescente e limitar
    candidates = sorted(candidates, key=lambda b: b[2] * b[3], reverse=True)[:4]
    return candidates

# -------------------- LÓGICA DE CONTAGEM E REGISTRO --------------------
# trackers:
last_positions = {}  # placa_text -> last y coordinate
active_plates_in_frame = set()
global_count = 0
global_count_lock = threading.Lock()

def handle_plate_detection(plate_text, centroid_y):
    """
    Decide entrada/saida baseado na travessia da linha COUNTING_LINE_Y.
    Se placa não existente ou sem registro aberto -> inserir entrada.
    Se houver registro sem saída -> atualizar saída.
    """
    global global_count
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # acessa last_positions
    prev_y = last_positions.get(plate_text)
    last_positions[plate_text] = centroid_y

    # detectar travessia: se prev_y existia e agora cruzou a linha
    crossed = False
    if prev_y is not None:
        # atravessou de cima para baixo
        if prev_y < COUNTING_LINE_Y <= centroid_y:
            crossed = True
        # atravessou de baixo para cima
        elif prev_y > COUNTING_LINE_Y >= centroid_y:
            crossed = True

    if crossed:
        # verificar se há um registro aberto (sem saída)
        # se não houver -> registrar entrada, incrementar contador
        # se houver -> registrar saída
        # Simplificação: se último registro sem saída existe -> marcar saída
        with db_lock:
            conn = sqlite3.connect(DB_FILENAME)
            cur = conn.cursor()
            cur.execute("""
                SELECT id FROM Registro_Passagens
                WHERE placa_veiculo = ? AND data_hora_saida IS NULL
                ORDER BY id DESC LIMIT 1
            """, (plate_text,))
            r = cur.fetchone()
            conn.close()

        if r is None:
            # entrada
            insert_passagem(plate_text, now)
            with global_count_lock:
                global_count += 1
            print(f"[{now}] Entrada registrada para {plate_text}")
        else:
            # saída
            update_saida(plate_text, now)
            print(f"[{now}] Saída registrada para {plate_text}")

# -------------------- THREAD DE CAPTURA E PROCESSAMENTO DE VÍDEO --------------------
stop_event = threading.Event()

def video_loop(app_ref):
    global active_plates_in_frame
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        messagebox.showerror("Erro", "Não foi possível abrir a câmera (verifique CAM_INDEX).")
        return

    # settings opcionais
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        candidates = detect_plate_candidates(frame)
        detected_this_frame = set()

        for (x, y, w, h) in candidates:
            roi = frame[y:y+h, x:x+w]
            plate_text = ocr_plate_from_roi(roi)
            if plausible_plate(plate_text):
                # centro y
                cy = y + h // 2
                detected_this_frame.add(plate_text)
                # desenhar bounding box e texto
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                # lógica de controle
                handle_plate_detection(plate_text, cy)

                # simular abertura se autorizado
                if is_authorized(plate_text):
                    # desenhar indicação de autorizado
                    cv2.putText(frame, "AUTORIZADO - ABRINDO", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                else:
                    cv2.putText(frame, "NAO AUTORIZADO", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # atualizar conjunto de placas ativas
        active_plates_in_frame = detected_this_frame

        # desenhar linha de contagem
        h_frame = frame.shape[0]
        cv2.line(frame, (0, COUNTING_LINE_Y), (frame.shape[1], COUNTING_LINE_Y), (255, 0, 0), 2)
        # exibir contador
        with global_count_lock:
            count_text = f"Total passada(s): {global_count}"
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        # quantos carros no frame (baseado no OCR detectado)
        cv2.putText(frame, f"Carros no frame: {len(active_plates_in_frame)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        # mostrar frame
        cv2.imshow("LPR - Controle de Acesso (Press q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------- GUI --------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Controle de Acesso Veicular - Protótipo")
        root.geometry("700x420")
        self.create_widgets()
        # start video thread
        self.video_thread = threading.Thread(target=video_loop, args=(self,), daemon=True)
        self.video_thread.start()
        # atualizar listagem periodicamente
        self.update_autorizadas_list()

    def create_widgets(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Cadastrar placa
        lbl = ttk.Label(frm, text="Cadastrar nova placa:")
        lbl.grid(column=0, row=0, sticky=tk.W)
        self.entry_placa = ttk.Entry(frm, width=20)
        self.entry_placa.grid(column=1, row=0, sticky=tk.W)
        btn_add = ttk.Button(frm, text="Adicionar", command=self.on_add_placa)
        btn_add.grid(column=2, row=0, sticky=tk.W)

        # Listagem autorizadas
        lbl2 = ttk.Label(frm, text="Placas autorizadas:")
        lbl2.grid(column=0, row=1, sticky=tk.W, pady=(10,0))
        self.listbox = tk.Listbox(frm, height=10, width=30)
        self.listbox.grid(column=0, row=2, rowspan=4, sticky=tk.W)

        # Botão abrir cancela manual
        lbl_manual = ttk.Label(frm, text="Abertura manual (digite a placa):")
        lbl_manual.grid(column=1, row=2, sticky=tk.W)
        self.entry_manual = ttk.Entry(frm, width=20)
        self.entry_manual.grid(column=1, row=3, sticky=tk.W)
        btn_manual = ttk.Button(frm, text="Abrir Cancela", command=self.on_manual_open)
        btn_manual.grid(column=2, row=3, sticky=tk.W)

        # Botões adicionais
        btn_refresh = ttk.Button(frm, text="Atualizar lista", command=self.update_autorizadas_list)
        btn_refresh.grid(column=1, row=4, sticky=tk.W, pady=(10,0))

        btn_show_log = ttk.Button(frm, text="Mostrar registro (console)", command=self.print_registros)
        btn_show_log.grid(column=1, row=5, sticky=tk.W, pady=(10,0))

        btn_stop = ttk.Button(frm, text="Parar sistema", command=self.on_stop)
        btn_stop.grid(column=2, row=5, sticky=tk.W)

    def on_add_placa(self):
        placa = self.entry_placa.get().strip().upper()
        placa = clean_plate_text(placa)
        if not placa:
            messagebox.showwarning("Aviso", "Digite uma placa válida.")
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

    def on_manual_open(self):
        placa = self.entry_manual.get().strip().upper()
        placa = clean_plate_text(placa)
        if not placa:
            messagebox.showwarning("Aviso", "Digite uma placa válida.")
            return
        # simula abertura: se autorizada, registra entrada
        if is_authorized(placa):
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insert_passagem(placa, now)
            with global_count_lock:
                global global_count
                global_count += 1
            messagebox.showinfo("Cancela", f"Cancela aberta para {placa} (manual).")
        else:
            messagebox.showwarning("Cancela", f"Placa {placa} não autorizada. Não abriu.")

    def print_registros(self):
        with db_lock:
            conn = sqlite3.connect(DB_FILENAME)
            cur = conn.cursor()
            cur.execute("SELECT id, placa_veiculo, data_hora_entrada, data_hora_saida, contador_total FROM Registro_Passagens ORDER BY id DESC LIMIT 50")
            rows = cur.fetchall()
            conn.close()
        print("===== Últimos registros =====")
        for r in rows:
            print(r)
        messagebox.showinfo("Registros", "Os últimas passagens foram exibidas no console.")

    def on_stop(self):
        if messagebox.askyesno("Confirmar", "Deseja parar o sistema (fechar câmera e sair)?"):
            stop_event.set()
            # aguarda thread e fecha app
            self.root.after(500, self.root.quit)

# -------------------- INICIALIZAÇÃO --------------------
def main():
    init_db()
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (stop_event.set(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()
