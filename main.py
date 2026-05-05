"""
main.py
=======
Interfaz Gráfica para Compresión de Imágenes mediante DCT
==========================================================
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from camera import capture_with_preview, is_camera_available
from dct_engine import (
    compress_dct,
    compute_psnr,
    get_compression_ratio,
    rgb_to_gray,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paleta de colores profesional (Inspirada en editores modernos / Dark Mode)
# ─────────────────────────────────────────────────────────────────────────────
BG_DARK      = "#121212"  # Fondo principal
BG_PANEL     = "#1E1E1E"  # Barras de herramientas
BG_CARD      = "#252526"  # Fondo de paneles
BG_CARD2     = "#2D2D30"  # Fondo de áreas de imagen
ACCENT       = "#007ACC"  # Azul profesional (Acción principal)
ACCENT2      = "#4FC1FF"  # Azul claro (Métricas)
ACCENT_HOVER = "#005C99"  # Hover principal
SUCCESS      = "#4CAF50"
WARNING      = "#FF9800"
DANGER       = "#F44336"
TEXT_PRIMARY = "#E0E0E0"
TEXT_MUTED   = "#9E9E9E"
TEXT_ACCENT  = "#4FC1FF"
BORDER       = "#333333"

FONT_TITLE  = ("Segoe UI", 16, "bold")
FONT_SUB    = ("Segoe UI", 10, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_METRIC = ("Segoe UI", 26, "bold")

CANVAS_W = 340
CANVAS_H = 260
MAX_COEFFS = 64  # 8×8


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de imagen para Tkinter
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_tk(pil_img: Image.Image, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    orig_w, orig_h = pil_img.size
    ratio = min(max_w / orig_w, max_h / orig_h)
    new_w = max(1, int(orig_w * ratio))
    new_h = max(1, int(orig_h * ratio))
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return ImageTk.PhotoImage(resized)


def array_to_pil_gray(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="L")


# ─────────────────────────────────────────────────────────────────────────────
# Componente: Panel de imagen con título
# ─────────────────────────────────────────────────────────────────────────────

class ImagePanel(tk.Frame):
    def __init__(self, parent, title: str, badge_color: str = ACCENT, **kwargs):
        super().__init__(parent, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1, **kwargs)
        self._title = title
        self._badge_color = badge_color
        self._tk_img = None 
        self._img_item = None
        self._build()

    def _build(self):
        header = tk.Frame(self, bg=BG_CARD, pady=8)
        header.pack(fill=tk.X, padx=12)

        badge = tk.Label(header, text="■", fg=self._badge_color, bg=BG_CARD, font=("Segoe UI", 10))
        badge.pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header, text=self._title, fg=TEXT_PRIMARY, bg=BG_CARD, font=FONT_SUB).pack(side=tk.LEFT)

        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.pack(fill=tk.X)

        self.canvas = tk.Canvas(
            self, bg=BG_CARD2, width=CANVAS_W, height=CANVAS_H, highlightthickness=0, bd=0
        )
        self.canvas.pack(padx=1, pady=1, fill=tk.BOTH, expand=True)

        self._placeholder = self.canvas.create_text(
            CANVAS_W // 2, CANVAS_H // 2, text="Sin imagen", fill=TEXT_MUTED, font=FONT_BODY
        )

        self.canvas.bind("<Configure>", self._on_resize)

        info_frame = tk.Frame(self, bg=BG_CARD, height=30)
        info_frame.pack(fill=tk.X)
        info_frame.pack_propagate(False)
        
        self.info_label = tk.Label(info_frame, text="—", fg=TEXT_MUTED, bg=BG_CARD, font=FONT_SMALL)
        self.info_label.pack(expand=True)

    def _on_resize(self, event):
        if self._tk_img is not None and self._img_item is not None:
            cx, cy = event.width // 2, event.height // 2
            self.canvas.coords(self._img_item, cx, cy)
        elif self._placeholder:
            cx, cy = event.width // 2, event.height // 2
            self.canvas.coords(self._placeholder, cx, cy)

    def set_image(self, pil_img: Image.Image, info: str = ""):
        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width()  or CANVAS_W
        ch = self.canvas.winfo_height() or CANVAS_H

        tk_img = pil_to_tk(pil_img.copy(), cw, ch)
        self._tk_img = tk_img

        self.canvas.delete("all")
        cx, cy = cw // 2, ch // 2
        self._img_item = self.canvas.create_image(cx, cy, anchor=tk.CENTER, image=self._tk_img)
        self.canvas.image = self._tk_img

        self.info_label.configure(text=info, fg=TEXT_MUTED)

    def clear(self):
        self._tk_img = None
        self._img_item = None
        if hasattr(self, 'canvas'):
            self.canvas.delete("all")
            self.canvas.image = None
            cw = self.canvas.winfo_width()  or CANVAS_W
            ch = self.canvas.winfo_height() or CANVAS_H
            self._placeholder = self.canvas.create_text(
                cw // 2, ch // 2, text="Sin imagen", fill=TEXT_MUTED, font=FONT_BODY
            )
        self.info_label.configure(text="—")


# ─────────────────────────────────────────────────────────────────────────────
# Componente: Tarjeta de métrica
# ─────────────────────────────────────────────────────────────────────────────

class MetricCard(tk.Frame):
    def __init__(self, parent, label: str, unit: str = "", **kwargs):
        super().__init__(parent, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1, padx=20, pady=10, **kwargs)
        self._unit = unit
        tk.Label(self, text=label.upper(), fg=TEXT_MUTED, bg=BG_CARD, font=("Segoe UI", 8, "bold")).pack()
        self.value_lbl = tk.Label(self, text="—", fg=TEXT_PRIMARY, bg=BG_CARD, font=FONT_METRIC)
        self.value_lbl.pack()
        if unit:
            tk.Label(self, text=unit, fg=TEXT_MUTED, bg=BG_CARD, font=FONT_SMALL).pack()

    def set(self, value: str, color: str = TEXT_PRIMARY):
        self.value_lbl.configure(text=value, fg=color)


# ─────────────────────────────────────────────────────────────────────────────
# Aplicación principal
# ─────────────────────────────────────────────────────────────────────────────

class DCTApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Sistema de Compresión DCT")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.minsize(1200, 760)

        self._color_img: Image.Image | None = None
        self._gray_arr: np.ndarray | None = None
        self._comp_arr: np.ndarray | None = None
        self._processing = False

        self._build_ui()
        self._center_window()

    # ------------------------------------------------------------------ #
    # Construcción de la UI                                              #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        self._build_header()
        self._build_toolbar()
        self._build_main_area()
        self._build_controls()
        self._build_statusbar()

    def _build_header(self):
        hdr = tk.Frame(self, bg=BG_PANEL, height=60, highlightbackground=BORDER, highlightthickness=1)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        tk.Label(
            hdr, text="Compresión de Imágenes DCT", font=FONT_TITLE, fg=TEXT_PRIMARY, bg=BG_PANEL
        ).pack(side=tk.LEFT, padx=20, pady=15)

        tk.Label(
            hdr, text="Procesamiento en Bloques 8×8 | Método Zig-Zag", font=FONT_BODY, fg=TEXT_MUTED, bg=BG_PANEL
        ).pack(side=tk.RIGHT, padx=20)

    def _build_toolbar(self):
        bar = tk.Frame(self, bg=BG_DARK, pady=16, padx=14)
        bar.pack(fill=tk.X)

        btn_cfg = [
            ("Capturar Cámara", self._btn_camera, BG_CARD2, "Captura foto con la webcam"),
            ("Cargar Imagen", self._btn_load, BG_CARD2, "Abre una imagen del disco"),
            ("Convertir a Grises", self._btn_convert, BG_CARD2, "Convierte a escala de grises"),
            ("Comprimir (DCT)", self._btn_compress, ACCENT, "Aplica compresión DCT"),
            ("Guardar Resultado", self._btn_save, BG_CARD2, "Guarda la imagen comprimida"),
        ]

        self._btns: dict[str, tk.Button] = {}
        for text, cmd, color, tip in btn_cfg:
            btn = tk.Button(
                bar, text=text, command=cmd, bg=color, fg=TEXT_PRIMARY, font=FONT_SUB,
                relief=tk.FLAT, padx=16, pady=8, cursor="hand2",
                activebackground=ACCENT_HOVER if color == ACCENT else BORDER,
                activeforeground=TEXT_PRIMARY, bd=1, highlightbackground=BORDER
            )
            btn.pack(side=tk.LEFT, padx=(0, 10))
            self._btns[text] = btn
            self._bind_hover(btn, color, ACCENT_HOVER if color == ACCENT else BORDER)

        cam_ok = is_camera_available()
        cam_color = SUCCESS if cam_ok else DANGER
        cam_text = "Estado: Cámara Detectada" if cam_ok else "Estado: Cámara No Detectada"
        tk.Label(bar, text=cam_text, fg=cam_color, bg=BG_DARK, font=FONT_SMALL).pack(side=tk.RIGHT, padx=10)

    def _build_main_area(self):
        main = tk.Frame(self, bg=BG_DARK)
        main.pack(fill=tk.BOTH, expand=True, padx=14)

        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.columnconfigure(2, weight=1)
        main.rowconfigure(0, weight=1)

        self._panel_orig  = ImagePanel(main, "Imagen Original", badge_color=ACCENT2)
        self._panel_mono  = ImagePanel(main, "Escala de Grises", badge_color=WARNING)
        self._panel_comp  = ImagePanel(main, "Resultado Comprimido", badge_color=SUCCESS)

        self._panel_orig.grid(row=0, column=0, padx=6, sticky="nsew")
        self._panel_mono.grid(row=0, column=1, padx=6, sticky="nsew")
        self._panel_comp.grid(row=0, column=2, padx=6, sticky="nsew")

    def _build_controls(self):
        ctrl = tk.Frame(self, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        ctrl.pack(fill=tk.X, padx=20, pady=20)

        left = tk.Frame(ctrl, bg=BG_PANEL)
        left.pack(side=tk.LEFT, padx=20, pady=15, fill=tk.BOTH, expand=True)

        tk.Label(left, text="Coeficientes DCT (Zig-Zag):", fg=TEXT_PRIMARY, bg=BG_PANEL, font=FONT_SUB).pack(anchor=tk.W)

        slider_row = tk.Frame(left, bg=BG_PANEL)
        slider_row.pack(fill=tk.X, pady=8)

        self._n_var = tk.IntVar(value=8)
        self._slider = ttk.Scale(
            slider_row, from_=1, to=MAX_COEFFS, orient=tk.HORIZONTAL, variable=self._n_var, command=self._on_slider
        )
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))

        self._coeff_lbl = tk.Label(
            slider_row, text="8 / 64", fg=ACCENT2, bg=BG_PANEL, font=("Segoe UI", 12, "bold"), width=7, anchor=tk.E
        )
        self._coeff_lbl.pack(side=tk.RIGHT)

        quick_row = tk.Frame(left, bg=BG_PANEL)
        quick_row.pack(anchor=tk.W, pady=(5, 0))
        tk.Label(quick_row, text="Ajustes Rápidos:", fg=TEXT_MUTED, bg=BG_PANEL, font=FONT_SMALL).pack(side=tk.LEFT, padx=(0, 10))
        
        for n in [1, 4, 8, 16, 32, 48, 64]:
            b = tk.Button(
                quick_row, text=str(n), width=4, command=lambda v=n: self._set_coeffs(v),
                bg=BG_CARD2, fg=TEXT_PRIMARY, font=FONT_SMALL, relief=tk.FLAT,
                cursor="hand2", activebackground=ACCENT
            )
            b.pack(side=tk.LEFT, padx=3)
            self._bind_hover(b, BG_CARD2, BORDER)

        metrics = tk.Frame(ctrl, bg=BG_PANEL)
        metrics.pack(side=tk.RIGHT, padx=20, pady=15)

        self._psnr_card  = MetricCard(metrics, "Calidad PSNR", "dB")
        self._psnr_card.pack(side=tk.LEFT, padx=8)

        self._ratio_card = MetricCard(metrics, "Tasa Descarte", "%")
        self._ratio_card.pack(side=tk.LEFT, padx=8)

        self._size_card  = MetricCard(metrics, "Resolución", "Píxeles")
        self._size_card.pack(side=tk.LEFT, padx=8)

    def _build_statusbar(self):
        self._status_var = tk.StringVar(value="Listo. Inicializando sistema...")
        bar = tk.Frame(self, bg=ACCENT, height=28)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)

        self._progress = ttk.Progressbar(bar, mode="indeterminate", length=150)
        self._progress.pack(side=tk.RIGHT, padx=10, pady=4)

        tk.Label(
            bar, textvariable=self._status_var, fg="#FFFFFF", bg=ACCENT, font=FONT_SMALL, anchor=tk.W
        ).pack(side=tk.LEFT, padx=10, fill=tk.X)

    # ------------------------------------------------------------------ #
    # Acciones de los botones                                            #
    # ------------------------------------------------------------------ #

    def _btn_camera(self):
        if not is_camera_available():
            messagebox.showerror("Error", "Cámara no disponible. Verifique la conexión.")
            return

        self._set_status("Iniciando captura de cámara...")
        self.update()

        def _capture():
            try:
                img = capture_with_preview()
                if img is not None:
                    self.after(0, lambda: self._set_color_image(img, "Captura Web"))
                else:
                    self.after(0, lambda: self._set_status("Captura cancelada por el usuario."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=_capture, daemon=True).start()

    def _btn_load(self):
        path = filedialog.askopenfilename(
            title="Seleccionar Archivo de Imagen",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("Todos los archivos", "*.*")]
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            name = os.path.basename(path)
            self._set_color_image(img, name)
        except Exception as e:
            messagebox.showerror("Error", f"Fallo al procesar el archivo:\n{e}")

    def _btn_convert(self):
        if self._color_img is None:
            messagebox.showwarning("Advertencia", "Requiere una imagen de origen.")
            return

        arr = np.array(self._color_img)
        self._gray_arr = rgb_to_gray(arr).astype(np.uint8)
        self._comp_arr = None
        self._panel_comp.clear()
        self._psnr_card.set("—")
        self._ratio_card.set("—")

        pil_gray = array_to_pil_gray(self._gray_arr)
        h, w = self._gray_arr.shape
        self._panel_mono.set_image(pil_gray, f"Dimensiones: {w}x{h}")
        self._size_card.set(f"{w}x{h}", color=TEXT_ACCENT)
        self._set_status("Conversión completada. Listo para aplicar compresión.")

    def _btn_compress(self):
        if self._gray_arr is None:
            messagebox.showwarning("Advertencia", "Requiere conversión previa a escala de grises.")
            return
        if self._processing:
            return

        n = self._n_var.get()
        self._start_processing(f"Procesando matriz DCT con {n} coeficientes...")

        def _work():
            try:
                comp = compress_dct(self._gray_arr.astype(np.float64), n)
                psnr = compute_psnr(self._gray_arr, comp)
                ratio = get_compression_ratio(n)
                self.after(0, lambda: self._show_compressed(comp, psnr, ratio, n))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error de Procesamiento", str(e)))
            finally:
                self.after(0, self._stop_processing)

        threading.Thread(target=_work, daemon=True).start()

    def _btn_save(self):
        if self._comp_arr is None:
            messagebox.showwarning("Advertencia", "No hay datos procesados para exportar.")
            return

        path = filedialog.asksaveasfilename(
            title="Exportar Resultado",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("BMP", "*.bmp"), ("JPEG", "*.jpg")]
        )
        if not path:
            return
        try:
            pil = array_to_pil_gray(self._comp_arr)
            pil.save(path)
            self._set_status(f"Exportación exitosa: {path}")
        except Exception as e:
            messagebox.showerror("Error de E/S", str(e))

    # ------------------------------------------------------------------ #
    # Lógica interna                                                     #
    # ------------------------------------------------------------------ #

    def _set_color_image(self, img: Image.Image, source: str):
        self._color_img = img
        self._gray_arr  = None
        self._comp_arr  = None
        self._panel_mono.clear()
        self._panel_comp.clear()
        self._psnr_card.set("—")
        self._ratio_card.set("—")

        w, h = img.size
        self._panel_orig.set_image(img, f"Origen: {source}")
        self._size_card.set(f"{w}x{h}", color=TEXT_ACCENT)
        self._set_status(f"Carga exitosa. Dimensiones base: {w}x{h} px.")

    def _show_compressed(self, comp: np.ndarray, psnr: float, ratio: float, n: int):
        self._comp_arr = comp
        pil = array_to_pil_gray(comp)
        
        psnr_str = f"{psnr:.2f}" if psnr != float('inf') else "MAX"
        psnr_color = SUCCESS if psnr > 30 else (WARNING if psnr > 20 else DANGER)

        self._panel_comp.set_image(pil, f"Coeficientes: {n}/64")
        self._psnr_card.set(psnr_str, color=psnr_color)
        self._ratio_card.set(f"{ratio:.1f}", color=TEXT_PRIMARY)
        self._set_status(f"Procesamiento finalizado. Descarte de datos: {ratio:.1f}%.")

    def _on_slider(self, val):
        n = int(float(val))
        self._coeff_lbl.configure(text=f"{n} / 64")

    def _set_coeffs(self, n: int):
        self._n_var.set(n)
        self._coeff_lbl.configure(text=f"{n} / 64")

    def _set_status(self, msg: str):
        self._status_var.set(msg)
        self.update_idletasks()

    def _start_processing(self, msg: str):
        self._processing = True
        self._btns["Comprimir (DCT)"].config(state=tk.DISABLED)
        self._set_status(msg)
        self._progress.start(12)

    def _stop_processing(self):
        self._processing = False
        self._btns["Comprimir (DCT)"].config(state=tk.NORMAL)
        self._progress.stop()

    @staticmethod
    def _bind_hover(widget: tk.Button, normal: str, hover: str):
        widget.bind("<Enter>", lambda e: widget.configure(bg=hover) if widget['state'] != tk.DISABLED else None)
        widget.bind("<Leave>", lambda e: widget.configure(bg=normal) if widget['state'] != tk.DISABLED else None)

    def _center_window(self):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w, h = 1200, 760
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")


# ─────────────────────────────────────────────────────────────────────────────
# Estilos ttk
# ─────────────────────────────────────────────────────────────────────────────

def _apply_ttk_styles():
    style = ttk.Style()
    style.theme_use("clam")
    
    style.configure(
        "TScale", background=BG_PANEL, troughcolor=BG_CARD, sliderlength=18, sliderrelief="flat"
    )
    style.map("TScale", background=[("active", BG_PANEL)])
    
    style.configure(
        "TProgressbar", troughcolor=BG_DARK, background="#FFFFFF", thickness=4, bordercolor=ACCENT, lightcolor="#FFFFFF", darkcolor="#FFFFFF"
    )


if __name__ == "__main__":
    app = DCTApp()
    _apply_ttk_styles()
    app.mainloop()