"""
experiments.py
==============
Script de Experimentos — Compresión DCT con diferentes coeficientes
====================================================================

Genera un informe visual comparativo para 4 imágenes (2 pre-almacenadas +
2 de cámara o adicionales), probando múltiples valores de coeficientes DCT.

Produce:
  - Figura matplotlib con paneles comparativos por imagen
  - Curva PSNR vs. número de coeficientes
  - Tabla resumen en consola
  - Archivo PDF/PNG del informe

Uso:
    python experiments.py
    python experiments.py --camara        # captura 2 imágenes con webcam
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("TkAgg")           # backend compatible con Tkinter en Windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from camera import capture_with_preview, is_camera_available
from dct_engine import compress_dct, compute_psnr, get_compression_ratio, rgb_to_gray

# ─────────────────────────────────────────────────────────────────────────────
# Configuración del experimento
# ─────────────────────────────────────────────────────────────────────────────

# Coeficientes a evaluar (cubriendo todo el rango 1–64)
COEFF_VALUES = [1, 2, 4, 6, 8, 10, 16, 20, 24, 32, 40, 48, 56, 64]

# Coeficientes de ejemplo para la visualización side-by-side
DEMO_COEFFS  = [1, 4, 8, 16, 32, 64]

# Imágenes pre-almacenadas (relativas al directorio del script)
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
PRELOADED_IMAGES = [
    ("assets/imagen_paisaje.png", "Paisaje (pre-almacenada)"),
    ("assets/imagen_retrato.png", "Retrato (pre-almacenada)"),
]

# Paleta de colores para las curvas
CURVE_COLORS = ["#6c63ff", "#22c55e", "#f59e0b", "#ef4444"]

# ─────────────────────────────────────────────────────────────────────────────
# Carga de imágenes
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Carga imagen RGB y convierte a escala de grises (uint8)."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    gray = rgb_to_gray(arr).astype(np.uint8)
    return gray


def capture_camera_images(n: int = 2) -> list[tuple[np.ndarray, str]]:
    """Captura n imágenes de la cámara. Retorna lista de (array, nombre)."""
    results = []
    if not is_camera_available():
        print("[AVISO] Camara no disponible. Se usaran imagenes sinteticas.")
        for i in range(n):
            synthetic = np.random.randint(30, 220, (256, 256), dtype=np.uint8)
            results.append((synthetic, f"Imagen sintetica {i+1}"))
        return results

    for i in range(n):
        print(f"\nCaptura {i+1}/{n}: Presiona ESPACIO para capturar, ESC para saltar.")
        img = capture_with_preview()
        if img is not None:
            arr = np.array(img)
            gray = rgb_to_gray(arr).astype(np.uint8)
            results.append((gray, f"Camara {i+1}"))
        else:
            print(f"  Captura {i+1} cancelada — usando imagen sintetica.")
            synthetic = np.random.randint(30, 220, (256, 256), dtype=np.uint8)
            results.append((synthetic, f"Sintetica {i+1}"))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experimento: calcular PSNR para todos los coeficientes
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(gray: np.ndarray) -> dict:
    """
    Comprime la imagen con cada valor de COEFF_VALUES y retorna un dict con:
      - psnr_curve: lista de PSNR en dB
      - demo_images: dict {n_coeff: array_comprimido} para DEMO_COEFFS
    """
    psnr_curve = []
    demo_images = {}

    for n in COEFF_VALUES:
        comp = compress_dct(gray.astype(np.float64), n)
        psnr = compute_psnr(gray, comp)
        psnr_curve.append(psnr if psnr != float('inf') else 100.0)

        if n in DEMO_COEFFS:
            demo_images[n] = comp

    return {"psnr_curve": psnr_curve, "demo_images": demo_images}


# ─────────────────────────────────────────────────────────────────────────────
# Visualización
# ─────────────────────────────────────────────────────────────────────────────

def plot_demo_row(axes_row, gray: np.ndarray, demo_images: dict, title: str, h: int, w: int):
    """Dibuja la fila de imágenes comprimidas para una imagen dada."""
    for ax, n in zip(axes_row, DEMO_COEFFS):
        if n == 64:
            img_show = gray
            ax.set_title(f"Original / 64 coef\n(∞ dB)", fontsize=7.5, color="#22c55e")
        else:
            img_show = demo_images[n]
            psnr = compute_psnr(gray, img_show)
            ratio = get_compression_ratio(n)
            ax.set_title(
                f"{n} coef  ({ratio:.0f}% desc.)\nPSNR={psnr:.1f} dB",
                fontsize=7.5, color="#a78bfa"
            )
        ax.imshow(img_show, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")

    axes_row[0].set_ylabel(f"{title}\n{w}×{h}px", fontsize=8,
                            rotation=0, labelpad=80, va='center',
                            color="#f1f5f9")


def build_report(images: list[tuple[np.ndarray, str]],
                  results: list[dict],
                  output_path: str = "informe_dct.png"):
    """Genera la figura completa del informe."""

    n_imgs = len(images)
    n_demo = len(DEMO_COEFFS)

    # ── Layout: n_imgs filas de demo + 1 fila de curvas PSNR ──────────────
    fig = plt.figure(figsize=(18, 4.5 * n_imgs + 5), facecolor="#0f1117")
    outer = gridspec.GridSpec(
        n_imgs + 1, 1,
        hspace=0.55,
        figure=fig,
    )

    # ── Título global ──────────────────────────────────────────────────────
    fig.suptitle(
        "Compresion de Imagenes mediante DCT  —  Bloques 8x8  —  Orden Zig-Zag",
        fontsize=15, fontweight="bold", color="#f1f5f9", y=0.98,
    )

    # ── Filas de imágenes de demo ──────────────────────────────────────────
    for row_idx, ((gray, name), res) in enumerate(zip(images, results)):
        h, w = gray.shape
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_demo, subplot_spec=outer[row_idx], wspace=0.05
        )
        axes = [fig.add_subplot(inner[0, col]) for col in range(n_demo)]
        for ax in axes:
            ax.set_facecolor("#1a1d27")
        plot_demo_row(axes, gray, res["demo_images"], name, h, w)

    # ── Curvas PSNR vs coeficientes ────────────────────────────────────────
    ax_psnr = fig.add_subplot(outer[n_imgs])
    ax_psnr.set_facecolor("#1a1d27")
    ax_psnr.spines[:].set_color("#2e3451")
    ax_psnr.tick_params(colors="#94a3b8")
    ax_psnr.xaxis.label.set_color("#94a3b8")
    ax_psnr.yaxis.label.set_color("#94a3b8")
    ax_psnr.set_title("Curva PSNR vs. Numero de Coeficientes DCT",
                       color="#f1f5f9", fontsize=12, pad=10)
    ax_psnr.set_xlabel("Numero de coeficientes (orden zig-zag)", fontsize=10)
    ax_psnr.set_ylabel("PSNR (dB)", fontsize=10)
    ax_psnr.set_xlim(0, 66)
    ax_psnr.set_xticks(COEFF_VALUES)
    ax_psnr.grid(True, color="#2e3451", linestyle="--", alpha=0.6)

    for (gray, name), res, color in zip(images, results, CURVE_COLORS):
        psnr_vals = res["psnr_curve"]
        ax_psnr.plot(
            COEFF_VALUES, psnr_vals,
            marker="o", markersize=5,
            linewidth=2, color=color, label=name,
        )
        # Anotación en el punto de 8 coeficientes
        idx8 = COEFF_VALUES.index(8)
        ax_psnr.annotate(
            f"{psnr_vals[idx8]:.1f}dB",
            (8, psnr_vals[idx8]),
            textcoords="offset points", xytext=(4, 6),
            fontsize=7, color=color,
        )

    # Líneas de referencia de calidad
    for db, label, lc in [(40, "Excelente (>40dB)", "#22c55e"),
                           (30, "Buena (>30dB)", "#f59e0b"),
                           (20, "Aceptable (>20dB)", "#ef4444")]:
        ax_psnr.axhline(db, color=lc, linestyle=":", alpha=0.5, linewidth=1)
        ax_psnr.text(65, db + 0.5, label, color=lc, fontsize=7, ha="right")

    ax_psnr.legend(
        facecolor="#1a1d27", edgecolor="#2e3451",
        labelcolor="#f1f5f9", fontsize=9,
    )

    plt.savefig(output_path, dpi=120, bbox_inches="tight",
                facecolor="#0f1117")
    print(f"\nInforme guardado en: {output_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tabla resumen en consola
# ─────────────────────────────────────────────────────────────────────────────

def print_table(images: list[tuple[np.ndarray, str]], results: list[dict]):
    """Imprime tabla de PSNR en la consola."""
    col_w = 14
    header_coefs = [str(n) for n in COEFF_VALUES]

    print("\n" + "="*90)
    print("  TABLA RESUMEN — PSNR (dB) por imagen y numero de coeficientes DCT")
    print("="*90)

    # Encabezado
    print(f"  {'Imagen':<28}", end="")
    for c in header_coefs:
        print(f"{c:>{col_w-2}}", end="  ")
    print()
    print("-"*90)

    for (gray, name), res in zip(images, results):
        print(f"  {name:<28}", end="")
        for psnr in res["psnr_curve"]:
            if psnr >= 100:
                print(f"{'inf':>{col_w-2}}", end="  ")
            else:
                print(f"{psnr:>{col_w-2}.1f}", end="  ")
        print()

    print("="*90)
    print(f"  Coeficientes evaluados: {COEFF_VALUES}")
    print(f"  Bloque DCT: 8x8  |  Metodo de seleccion: Zig-Zag")
    print("="*90 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experimentos de compresion DCT con diferentes coeficientes"
    )
    parser.add_argument(
        "--camara", action="store_true",
        help="Captura 2 imagenes adicionales con la webcam"
    )
    parser.add_argument(
        "--salida", default="informe_dct.png",
        help="Nombre del archivo de salida (default: informe_dct.png)"
    )
    args = parser.parse_args()

    print("="*60)
    print("  Experimentos DCT — Compresion de Imagenes 8x8 Zig-Zag")
    print("="*60)

    # ── Cargar imágenes pre-almacenadas ─────────────────────────────────
    images: list[tuple[np.ndarray, str]] = []
    for path, label in PRELOADED_IMAGES:
        full_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(full_path):
            print(f"[+] Cargando: {label} ({path})")
            gray = load_image(full_path)
            images.append((gray, label))
        else:
            print(f"[!] No encontrada: {full_path}  — usando imagen sintetica.")
            synthetic = np.random.randint(20, 230, (256, 256), dtype=np.uint8)
            images.append((synthetic, f"{label} (sintetica)"))

    # ── Imágenes de cámara (o sintéticas si no hay cámara) ───────────────
    if args.camara:
        cam_imgs = capture_camera_images(2)
    else:
        print("\n[INFO] Usando imagenes sinteticas para 'camara'.")
        print("       Ejecuta con --camara para capturar con la webcam.\n")
        # Imagen 3: gradiente (simula foto de interior)
        gx = np.linspace(0, 255, 320).reshape(1, -1).repeat(240, axis=0)
        gy = np.linspace(0, 180, 240).reshape(-1, 1).repeat(320, axis=1)
        noise = np.random.normal(0, 15, (240, 320))
        synth3 = np.clip(0.5 * gx + 0.5 * gy + noise, 0, 255).astype(np.uint8)
        # Imagen 4: textura (simula foto de exterior con detalles finos)
        xx, yy = np.meshgrid(np.linspace(0, 8*np.pi, 320),
                              np.linspace(0, 6*np.pi, 240))
        synth4 = np.clip(
            128 + 60*np.sin(xx)*np.cos(yy) + 40*np.sin(2*xx + yy)
            + np.random.normal(0, 10, (240, 320)),
            0, 255
        ).astype(np.uint8)
        cam_imgs = [(synth3, "Camara 1 (gradiente)"),
                    (synth4, "Camara 2 (textura)")]

    images.extend(cam_imgs)

    # ── Ejecutar experimentos ─────────────────────────────────────────────
    results = []
    for i, (gray, name) in enumerate(images):
        print(f"Procesando imagen {i+1}/4: {name}  ({gray.shape[1]}x{gray.shape[0]}px)")
        res = run_experiment(gray)
        results.append(res)
        psnr_8 = res["psnr_curve"][COEFF_VALUES.index(8)]
        psnr_32 = res["psnr_curve"][COEFF_VALUES.index(32)]
        print(f"   PSNR con  8 coef: {psnr_8:.2f} dB")
        print(f"   PSNR con 32 coef: {psnr_32:.2f} dB")

    # ── Tabla resumen ─────────────────────────────────────────────────────
    print_table(images, results)

    # ── Figura del informe ────────────────────────────────────────────────
    out = os.path.join(os.path.dirname(__file__), args.salida)
    fig = build_report(images, results, output_path=out)

    plt.show()


if __name__ == "__main__":
    main()
