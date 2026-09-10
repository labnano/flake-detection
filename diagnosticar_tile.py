"""
diagnosticar_tile.py  --  por que ESTE flake nao foi detectado?

Quando um flake que voce ve na imagem nao aparece na saida, ha tres explicacoes
possiveis, e elas pedem consertos completamente diferentes:

  1. O SEGMENTADOR nunca propos.  O M2F nao desenhou mascara nenhuma ali.
     Nenhum pos-filtro alcanca isso -- o conserto e re-treinar/reescalar o M2F.
  2. O VETO matou.  Foi proposto, mas o classificador julgou os pixels como
     substrato. Conserto: mexer no LIMIAR_MAHALANOBIS (parametros.py).
  3. O POS-FILTRO cortou.  Foi proposto, sobreviveu ao veto, mas o modelo deu
     probabilidade baixa. Conserto: feature nova, ou mais orcamento.

Este script separa os tres casos numa imagem so. Cada candidato do tile ganha
um contorno colorido:

    VERDE     aceito (prob >= LIMIAR_PROB_FLAKE)
    AMARELO   proposto e vivo, mas o pos-filtro cortou       -> caso 3
    VERMELHO  morto pelo veto do classificador               -> caso 2
    CINZA     abaixo do piso de score/size do pipeline
    (nada)    o segmentador nao viu                          -> caso 1

Se o seu monolayer nao tem contorno NENHUM, o problema esta no segmentador e
nao adianta mexer em limiar ou feature.

Uso:
    python diagnosticar_tile.py <caminho/da/imagem.tif> [--saida diag.png]
"""

import argparse
import os

import cv2
import numpy as np

from extrair_candidatos_v2 import analisar
from modelo import obter_predictor, obter_filtro
from parametros import (LIMIAR_PROB_FLAKE, LIMIAR_MAHALANOBIS,
                        SCORE_THRESHOLD, SIZE_THRESHOLD)

CORES = {
    "aceito":     (0, 255, 0),        # verde
    "pos-filtro": (0, 255, 255),      # amarelo
    "veto":       (0, 0, 255),        # vermelho
    "piso":       (150, 150, 150),    # cinza
}


def nome_limiar(t):
    """Mesma convencao do extrator: o limiar vira o sufixo da coluna."""
    return "inf" if t == float("inf") else str(int(t))


def main(caminho, saida):
    imagem = cv2.imread(caminho)
    if imagem is None:
        print(f"nao consegui ler: {caminho}")
        return

    predictor = obter_predictor()
    _, candidatos = analisar(predictor, imagem)
    filtro = obter_filtro()

    chave = nome_limiar(LIMIAR_MAHALANOBIS)
    marcada = imagem.copy()
    linhas = []

    for cand in candidatos:
        # Reproduz, na ordem, o que o pipeline faz com este candidato.
        if cand["score"] <= SCORE_THRESHOLD or cand["size_px"] <= SIZE_THRESHOLD:
            destino, prob = "piso", None
        elif cand["por_limiar"].get(chave, (0, 0))[0] == 0:
            destino, prob = "veto", None
        elif filtro is None:
            destino, prob = "aceito", None
        else:
            # As derivadas nao vem do extrator: sao calculadas na hora, com as
            # MESMAS formulas do treinar_filtro.carregar(). Manter as tres
            # copias (treino, inferencia, diagnostico) em sincronia e a parte
            # chata disso -- se divergirem, ninguem recebe erro, so numeros
            # diferentes de quem decide de verdade.
            for canal in "bgr":
                if f"std_{canal}" in cand:
                    cand[f"snr_{canal}"] = (abs(cand[f"c_{canal}"])
                                            / max(cand[f"std_{canal}"], 1e-4))
            cand["mag"] = (cand["c_b"] ** 2 + cand["c_g"] ** 2
                           + cand["c_r"] ** 2) ** 0.5

            faltando = [n for n in filtro["features"] if n not in cand]
            if faltando:
                print(f"[erro] o modelo pede features que o extrator nao "
                      f"produziu: {', '.join(faltando)}")
                return

            X = np.array([[cand[n] for n in filtro["features"]]])
            prob = float(filtro["modelo"].predict_proba(X)[0, 1])
            destino = "aceito" if prob >= LIMIAR_PROB_FLAKE else "pos-filtro"

        pts = np.array(cand["contorno"].split(), dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(marcada, [pts], -1, CORES[destino], 2)
        linhas.append((destino, prob, cand))

    print(f"{os.path.basename(caminho)}: {len(candidatos)} candidatos propostos "
          f"pelo segmentador\n")
    print(f"{'destino':<12}{'prob':>7}{'score':>8}{'size':>7}"
          f"{'c_b':>8}{'c_g':>8}{'c_r':>8}{'frac_sub':>10}")
    print("-" * 68)
    ordem = {"aceito": 0, "pos-filtro": 1, "veto": 2, "piso": 3}
    for destino, prob, c in sorted(linhas, key=lambda t: (ordem[t[0]], -(t[1] or 0))):
        p = f"{prob:.3f}" if prob is not None else "  --"
        print(f"{destino:<12}{p:>7}{c['score']:>8.3f}{c['size_px']:>7}"
              f"{c['c_b']:>8.3f}{c['c_g']:>8.3f}{c['c_r']:>8.3f}"
              f"{c['frac_substrato']:>10.3f}")

    contagem = {}
    for destino, _, _ in linhas:
        contagem[destino] = contagem.get(destino, 0) + 1
    print("\nresumo:", ", ".join(f"{k}={v}" for k, v in sorted(contagem.items())))
    print("VERDE=aceito  AMARELO=cortado pelo pos-filtro  "
          "VERMELHO=morto pelo veto  CINZA=abaixo do piso")
    print("Sem contorno nenhum sobre o seu flake = o segmentador nao o propos.")

    cv2.imwrite(saida, marcada)
    print(f"\nimagem: {saida}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("imagem", help="um .tif da varredura")
    p.add_argument("--saida", default=None, help="PNG de saida")
    a = p.parse_args()
    main(a.imagem, a.saida or os.path.splitext(os.path.basename(a.imagem))[0] + "_diag.png")
