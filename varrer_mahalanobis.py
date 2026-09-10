"""
varrer_mahalanobis.py  --  mede o efeito do LIMIAR_MAHALANOBIS sobre o veto do
classificador, sem re-rodar a parte cara do modelo para cada valor.

Mesmo truque do extrair_candidatos.py: o passo caro (segmentação + mapa de
distâncias) não depende do limiar. Roda-se uma vez por imagem, guarda-se o mapa,
e cada limiar vira uma comparação em cima dele.

Uso:
    python varrer_mahalanobis.py <pasta com .tif> [--n 120]
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch

from modelo import obter_predictor
from parametros import LIMIAR_CONTRASTE_G

# 10 = o default de hoje, a linha de base do experimento.
LIMIARES = [10, 12, 15, 20, 25, 30, 40, 60]


def imread_unicode(caminho):
    """Lê imagem mesmo com acento no caminho (mesmo truque do extrair_candidatos)."""
    return cv2.imdecode(np.fromfile(caminho, dtype=np.uint8), cv2.IMREAD_COLOR)


@torch.inference_mode()
def medir_imagem(predictor, imagem_bgr):
    """Roda o caro uma vez; devolve (n_candidatos, {limiar: (pos_veto, pos_cor)})."""
    imagem = torch.tensor(imagem_bgr, dtype=torch.uint8).to(predictor.device)

    # (1) CARO -- segmentação. filter_instances já aplica SCORE e SIZE, com os
    #     valores que estão no parametros.py. Não depende do limiar.
    instancias = predictor.filter_instances(predictor.segmentation_model(imagem))
    n_cand = len(instancias)
    if n_cand == 0:
        return 0, {lim: (0, 0) for lim in LIMIARES}

    # (2) CARO -- contraste e distâncias. get_mh_distance_map devolve (K, H, W):
    #     a distância de cada pixel a cada uma das K componentes do modelo.
    contraste = predictor.calculate_contrast_image(imagem)
    mapa = predictor.classification_model.get_mh_distance_map(contraste.float())
    dist, classe = mapa.min(dim=0)      # componente mais próxima, por pixel

    # (3) BARATO -- cada limiar é só uma comparação sobre `dist`.
    saida = {}
    for lim in LIMIARES:
        sem_seg = torch.where(dist > lim, torch.zeros_like(classe), classe)

        pos_veto = pos_cor = 0
        for i in range(n_cand):
            mascara = instancias[i].pred_masks.squeeze()
            classes, contagens = torch.unique(sem_seg[mascara], return_counts=True)
            if int(classes[torch.argmax(contagens)].item()) == 0:
                continue                       # <- o veto
            pos_veto += 1
            # contraste médio no verde, como o filtrar_por_cor faz
            if float(contraste[mascara][:, 1].mean().item()) <= LIMIAR_CONTRASTE_G:
                pos_cor += 1

        saida[lim] = (pos_veto, pos_cor)

    return n_cand, saida


def main(pasta, n_imagens):
    predictor = obter_predictor()

    todos = sorted(glob.glob(os.path.join(pasta, "**", "*.tif"), recursive=True))
    if not todos:
        print(f"Nenhuma imagem .tif em: {pasta}")
        return

    # Amostragem regular ao longo da varredura, não as N primeiras: as primeiras
    # posições caem todas na mesma região da amostra e não a representam.
    caminhos = todos[:: max(1, len(todos) // n_imagens)][:n_imagens]
    print(f"{len(caminhos)} imagens amostradas de {len(todos)}\n")

    total = 0
    veto = {lim: 0 for lim in LIMIARES}
    cor = {lim: 0 for lim in LIMIARES}
    posicoes = {lim: 0 for lim in LIMIARES}

    for k, caminho in enumerate(caminhos, 1):
        imagem = imread_unicode(caminho)
        if imagem is None:
            print(f"[erro] não li: {caminho}")
            continue
        n_cand, saida = medir_imagem(predictor, imagem)
        total += n_cand
        for lim, (nv, nc) in saida.items():
            veto[lim] += nv
            cor[lim] += nc
            posicoes[lim] += 1 if nc else 0
        if k % 20 == 0:
            print(f"  ... {k}/{len(caminhos)}")

    print(f"\ncandidatos que passam SCORE e SIZE: {total}")
    print(f"(o veto e o filtro de cor agem sobre estes)\n")
    print(f"{'limiar':>7} {'passam o veto':>15} {'%':>7} {'+ filtro de cor':>17} "
          f"{'posições c/ flake':>19}")
    for lim in LIMIARES:
        marca = "  <- hoje" if lim == 10 else ""
        print(f"{lim:>7} {veto[lim]:>15} {100 * veto[lim] / total if total else 0:>6.1f}% "
              f"{cor[lim]:>17} {posicoes[lim]:>13}/{len(caminhos)}{marca}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pasta", help="Pasta com as imagens .tif")
    p.add_argument("--n", type=int, default=120, help="Quantas imagens amostrar")
    args = p.parse_args()
    main(args.pasta, args.n)