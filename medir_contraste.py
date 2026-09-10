"""
medir_contraste.py  --  mede o contraste optico medio (BGR) de cada candidato,
OFFLINE, sem re-rodar o modelo.

Ideia
-----
O que distingue grafeno de cola quando os dois sao homogeneos e a ASSINATURA
OPTICA: o quanto o material escurece/colore em relacao ao substrato. O MaskTerial
mede isso como
    contraste = pixel / cor_de_fundo - 1      (por canal B, G, R)
    mean_contrast = media do contraste sobre a mascara do flake

A "cor de fundo" e o valor mais comum de cada canal (o substrato ocupa a maior
parte da imagem) -- o metodo calculate_mean_background_values do MaskTerial.

Aqui a gente reproduz isso a partir da IMAGEM + do CONTORNO ja gravado no
candidatos.csv -- nao precisa de GPU nem do modelo. A unica aproximacao e que o
contorno esta simplificado (poucos vertices), entao a mascara reconstruida e
levemente aproximada; para uma MEDIA sobre a regiao, isso e irrelevante.

Saida: contraste.csv (cand_id, c_b, c_g, c_r) ao lado do candidatos.csv.

Uso:
    python medir_contraste.py <caminho/para/candidatos.csv>
"""

import argparse
import csv
import os

import cv2
import numpy as np

# Mesmos parametros do calculate_mean_background_values do MaskTerial.
MIN_V, MAX_V, RAIO = 20, 230, 5


def imread_unicode(caminho):
    return cv2.imdecode(np.fromfile(caminho, np.uint8), cv2.IMREAD_COLOR)


def cor_de_fundo(img):
    """Cor do substrato por canal: media dos pixels em torno do modo do histograma."""
    means = np.zeros(3, dtype=np.float64)
    for c in range(3):
        canal = img[:, :, c]
        hist = np.bincount(canal.ravel(), minlength=256)
        modo = int(np.argmax(hist[MIN_V:MAX_V + 1])) + MIN_V      # valor mais comum
        lo = min(max(modo - RAIO, MIN_V), MAX_V)
        hi = min(max(modo + RAIO, MIN_V), MAX_V)
        sel = (canal >= lo) & (canal <= hi)
        means[c] = canal[sel].mean() if sel.any() else 1.0
    return means


def main(csv_path):
    base = os.path.dirname(csv_path)
    pasta_imagens = os.path.dirname(base)   # <mosaico>
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    rows.sort(key=lambda r: r["imagem"])    # agrupa por imagem -> abre cada uma 1x

    saida = os.path.join(base, "contraste.csv")
    cache_nome, img, means = None, None, None
    n = 0
    with open(saida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cand_id", "c_b", "c_g", "c_r"])

        for r in rows:
            if r["imagem"] != cache_nome:
                img = imread_unicode(os.path.join(pasta_imagens, r["imagem"]))
                cache_nome = r["imagem"]
                means = cor_de_fundo(img) if img is not None else None
            if img is None:
                continue

            nums = r["contorno"].split()
            if len(nums) < 6:               # < 3 vertices: contorno degenerado
                continue
            pts = np.array(nums, dtype=np.int32).reshape(-1, 2)

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            m = mask.astype(bool)
            if not m.any():
                continue

            # contraste = pixel/fundo - 1, so nos pixels da mascara; media BGR.
            pixels = img[m].astype(np.float32)
            c = (pixels / means - 1.0).mean(axis=0)
            w.writerow([r["cand_id"], f"{c[0]:.5f}", f"{c[1]:.5f}", f"{c[2]:.5f}"])
            n += 1

    print(f"medido: {n} candidatos")
    print(f"salvo:  {saida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="Caminho para o candidatos.csv")
    args = parser.parse_args()
    main(os.path.abspath(args.csv))
