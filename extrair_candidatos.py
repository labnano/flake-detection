"""
extrair_candidatos.py  --  PASSO 1 de 3 (versao definitiva).

O que faz
---------
Roda as redes do MaskTerial UMA vez por imagem e grava, num unico CSV, cada
candidato a flake com as grandezas que os limiares comparam
(score, size, max_prob) MAIS o contorno da forma detectada. O sweep (passo 3)
reproduz offline qualquer combinacao de limiares a partir dai, sem GPU.

Por que esta versao e diferente da primeira
--------------------------------------------
1) NAO salva crops. A primeira versao salvava um PNG por candidato -- numa
   varredura completa isso virava >150 mil arquivos. Agora o anotador (passo 2)
   recorta da imagem original na hora; nada de crops em disco.
2) Grava o CONTORNO da forma (poligono simplificado), nao so o bounding box.
   Assim o anotador desenha exatamente o que o modelo segmentou.
3) Aplica um PISO de score na gravacao: candidatos abaixo dele (ruido, ~96% do
   total) nem entram no CSV. O trabalho caro (rodar as redes) e o mesmo; o que
   muda e so nao escrever o lixo.

Uso:
    python extrair_candidatos.py <pasta_das_imagens> [--saida <pasta_de_saida>]
    (sem a pasta, abre uma janela para escolher)
"""

import argparse
import csv
import glob
import os
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import torch

from caminhos import DIRETORIO_SAIDA_BASE
from modelo import obter_predictor

# Piso de score na GRAVACAO: candidatos abaixo disso nem viram linha no CSV.
# 0.015 = o mesmo piso que a gente vai anotar. Abaixo disso e ruido puro.
PISO_EXTRACAO = 0.015

# Simplificacao do contorno (cv2.approxPolyDP): fracao do perimetro usada como
# tolerancia. 0.01 (1%) troca centenas de pontos por ~5-15, preservando a forma.
EPSILON_FRAC = 0.01


def imread_unicode(caminho):
    """Le uma imagem mesmo se o caminho tiver acento / caractere nao-ASCII.

    O cv2.imread usa a codepage ANSI do Windows e devolve None em caminhos com
    'o', 'a' etc. Aqui a gente le os bytes com np.fromfile -- que entende
    caminho Unicode -- e decodifica com cv2.imdecode.
    """
    dados = np.fromfile(caminho, dtype=np.uint8)
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


@torch.inference_mode()
def extrair_candidatos_crus(predictor, imagem_bgr):
    """Roda as redes e devolve os candidatos com score >= PISO_EXTRACAO.

    Replica o miolo de _inference_with_segmentation_model + infer_classes do
    MaskTerial, mas sem aplicar os limiares finais e devolvendo tambem o
    max_prob (ocupacao) e o contorno da forma.
    """
    imagem = torch.tensor(imagem_bgr, dtype=torch.uint8).to(predictor.device)

    # (1) SEGMENTACAO -- passo caro (GPU). Mascaras booleanas + score por candidato.
    instances = predictor.segmentation_model(imagem)
    if len(instances) == 0:
        return []

    # (2) CLASSIFICACAO -- mapa semantico de espessura, calculado uma vez.
    contrast_image = predictor.calculate_contrast_image(imagem)
    sem_seg = predictor.classification_model(contrast_image)

    candidatos = []
    for i in range(len(instances)):
        mask = instances[i].pred_masks.squeeze()
        score = instances[i].scores.squeeze().item()

        # Corta o ruido cedo: nao gasta tempo com occupancy/contorno de lixo.
        if score < PISO_EXTRACAO:
            continue

        # size = numero de pixels da mascara (o que o SIZE_THRESHOLD compara).
        size = int(torch.count_nonzero(mask).item())

        # max_prob = ocupacao da classe dominante (o que MIN_CLASS_OCCUPANCY compara).
        class_preds = sem_seg[mask]
        if class_preds.numel() == 0:
            continue
        classes, counts = torch.unique(class_preds, return_counts=True)
        normalizado = counts / torch.sum(counts)
        max_prob, max_index = torch.max(normalizado, dim=0)
        pred_class = int(classes[max_index].item())

        # Contorno da forma: maior contorno externo da mascara, simplificado.
        mask_np = (mask.cpu().numpy() > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        maior = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(maior)
        eps = EPSILON_FRAC * cv2.arcLength(maior, True)
        aprox = cv2.approxPolyDP(maior, eps, True).reshape(-1, 2)
        # Serializa "x1 y1 x2 y2 ..." (coords globais da imagem, separadas por espaco).
        contorno_str = " ".join(str(int(v)) for v in aprox.flatten())

        candidatos.append({
            "score": float(score),
            "size": size,
            "max_prob": float(max_prob.item()),
            "pred_class": pred_class,
            "x": x, "y": y, "w": w, "h": h,
            "contorno": contorno_str,
        })

    return candidatos


def main(pasta_imagens, pasta_saida):
    predictor = obter_predictor()

    caminhos = glob.glob(os.path.join(pasta_imagens, "**", "*.tif"), recursive=True)
    if not caminhos:
        print(f"Nenhuma imagem .tif encontrada em: {pasta_imagens}")
        return

    os.makedirs(pasta_saida, exist_ok=True)
    caminho_csv = os.path.join(pasta_saida, "candidatos.csv")

    total = 0
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["cand_id", "imagem", "score", "size", "max_prob",
             "pred_class", "x", "y", "w", "h", "contorno"]
        )

        for caminho in caminhos:
            imagem = imread_unicode(caminho)
            if imagem is None:
                print(f"[Erro] Nao consegui ler: {caminho}")
                continue

            candidatos = extrair_candidatos_crus(predictor, imagem)
            print(f"{os.path.basename(caminho)}: {len(candidatos)} candidato(s) >= {PISO_EXTRACAO}")

            rel = os.path.relpath(caminho, pasta_imagens)
            for cand in candidatos:
                writer.writerow([
                    total, rel,
                    f"{cand['score']:.6f}", cand["size"], f"{cand['max_prob']:.6f}",
                    cand["pred_class"], cand["x"], cand["y"], cand["w"], cand["h"],
                    cand["contorno"],
                ])
                total += 1

    print(f"\nPronto. {total} candidatos (score >= {PISO_EXTRACAO}) de {len(caminhos)} imagens.")
    print(f"CSV: {caminho_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pasta", nargs="?", default=None,
                        help="Pasta com as imagens .tif (se omitida, abre uma janela)")
    parser.add_argument("--saida", default=None,
                        help="Pasta de saida (default: <pasta_escolhida>/avaliacao_parametros)")
    args = parser.parse_args()

    pasta = args.pasta
    if pasta is None:
        raiz = tk.Tk()
        raiz.withdraw()
        pasta = filedialog.askdirectory(
            title="Escolha a pasta com as imagens .tif",
            initialdir=DIRETORIO_SAIDA_BASE,
        )
        raiz.destroy()
        if not pasta:
            print("Nenhuma pasta escolhida.")
            raise SystemExit

    # So agora, com 'pasta' resolvida, definimos a saida (dentro da pasta escolhida).
    saida = args.saida
    if saida is None:
        saida = os.path.join(pasta, "avaliacao_parametros")

    main(pasta, saida)
