"""
dct_engine.py
=============
Motor de compresión de imágenes mediante DCT (Discrete Cosine Transform).

Funcionalidades:
- Conversión a escala de grises
- DCT/IDCT por bloques 8×8
- Selección de coeficientes por método zig-zag
- Cálculo de PSNR
"""

import numpy as np
from scipy.fft import dctn, idctn


# ---------------------------------------------------------------------------
# Conversión de color
# ---------------------------------------------------------------------------

def rgb_to_gray(img_array: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen RGB (H×W×3) a escala de grises (H×W) usando los
    pesos estándar BT.601.
    """
    if img_array.ndim == 2:
        return img_array.astype(np.float64)
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.float64)


# ---------------------------------------------------------------------------
# Utilidades de relleno
# ---------------------------------------------------------------------------

def pad_image(img: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Rellena la imagen con ceros (padding) para que sus dimensiones sean
    múltiplos exactos de block_size.
    """
    h, w = img.shape
    new_h = int(np.ceil(h / block_size)) * block_size
    new_w = int(np.ceil(w / block_size)) * block_size
    padded = np.zeros((new_h, new_w), dtype=np.float64)
    padded[:h, :w] = img
    return padded


# ---------------------------------------------------------------------------
# Generación del orden zig-zag para un bloque 8×8
# ---------------------------------------------------------------------------

def zigzag_indices(block_size: int = 8) -> list:
    """
    Genera la lista de índices (fila, columna) en orden zig-zag para un
    bloque de tamaño block_size × block_size.

    Retorna:
        Lista de tuplas (row, col) en orden zig-zag.
    """
    indices = []
    for diag in range(2 * block_size - 1):
        if diag % 2 == 0:
            # Recorrido hacia arriba-derecha
            r_start = min(diag, block_size - 1)
            c_start = diag - r_start
            while r_start >= 0 and c_start < block_size:
                indices.append((r_start, c_start))
                r_start -= 1
                c_start += 1
        else:
            # Recorrido hacia abajo-izquierda
            c_start = min(diag, block_size - 1)
            r_start = diag - c_start
            while c_start >= 0 and r_start < block_size:
                indices.append((r_start, c_start))
                r_start += 1
                c_start -= 1
    return indices


# Precalcular los índices zig-zag para 8×8 (constante global)
ZIGZAG_8x8 = zigzag_indices(8)


# ---------------------------------------------------------------------------
# DCT 2D por bloque
# ---------------------------------------------------------------------------

def dct2_block(block: np.ndarray) -> np.ndarray:
    """Aplica DCT-II 2D ortogonal a un bloque."""
    return dctn(block, norm='ortho')


def idct2_block(block: np.ndarray) -> np.ndarray:
    """Aplica IDCT-II 2D ortogonal a un bloque."""
    return idctn(block, norm='ortho')


# ---------------------------------------------------------------------------
# Compresión principal
# ---------------------------------------------------------------------------

def compress_dct(gray_img: np.ndarray, n_coeffs: int,
                 block_size: int = 8) -> np.ndarray:
    """
    Comprime una imagen en escala de grises usando DCT por bloques de 8×8.

    Parámetros:
        gray_img  : Array 2D (float) en escala de grises [0, 255].
        n_coeffs  : Número de coeficientes DCT a conservar (orden zig-zag).
                    Rango válido: 1 – block_size².
        block_size: Tamaño del bloque (default 8).

    Retorna:
        Array 2D reconstruido (mismas dimensiones que gray_img), dtype uint8.
    """
    n_coeffs = max(1, min(n_coeffs, block_size * block_size))
    orig_h, orig_w = gray_img.shape

    # Padding
    padded = pad_image(gray_img, block_size)
    h, w = padded.shape

    # Máscara zig-zag: solo los primeros n_coeffs índices quedan activos
    zigzag = ZIGZAG_8x8
    mask = np.zeros((block_size, block_size), dtype=np.float64)
    for k in range(n_coeffs):
        r, c = zigzag[k]
        mask[r, c] = 1.0

    reconstructed = np.zeros_like(padded, dtype=np.float64)

    # Procesar cada bloque
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = padded[i:i + block_size, j:j + block_size]
            # Centrar en [-128, 127] (común en JPEG)
            block_centered = block - 128.0
            # DCT
            coeffs = dct2_block(block_centered)
            # Aplicar máscara zig-zag
            coeffs_masked = coeffs * mask
            # IDCT
            block_rec = idct2_block(coeffs_masked)
            # Restaurar nivel de DC
            block_rec += 128.0
            reconstructed[i:i + block_size, j:j + block_size] = block_rec

    # Recortar al tamaño original y clipear a [0, 255]
    result = reconstructed[:orig_h, :orig_w]
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# Métricas de calidad
# ---------------------------------------------------------------------------

def compute_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calcula el PSNR (Peak Signal-to-Noise Ratio) en dB entre dos imágenes
    en escala de grises.

    PSNR = 10 · log₁₀(MAX² / MSE)   donde MAX = 255.

    Retorna:
        PSNR en dB. Retorna inf si las imágenes son idénticas.
    """
    orig = original.astype(np.float64)
    comp = compressed.astype(np.float64)
    mse = np.mean((orig - comp) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((255.0 ** 2) / mse)


def get_compression_ratio(n_coeffs: int, block_size: int = 8) -> float:
    """
    Retorna la relación de compresión teórica (coeficientes descartados /
    coeficientes totales).
    """
    total = block_size * block_size
    return (total - n_coeffs) / total * 100.0
