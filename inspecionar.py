"""
inspecionar.py  --  olha um subconjunto de candidatos e o MOTIVO de cada um.

Uso tipico: ver os flakes que uma regra deixa de fora (falsos negativos), para
decidir se o preco (flakes perdidos) vale o ganho (lixo removido).

Reaproveita a visualizacao do revisor.py (3 paineis: recorte com contorno,
recorte limpo, imagem inteira com o candidato destacado).

    MODO = "fn"  -> FLAKES rejeitados pela regra (o que voce pediu)
    MODO = "fp"  -> LIXO que a regra aceita (falsos positivos que sobram)

Navegacao (so olhar, nao altera nada):
    espaco / n = proximo    b = anterior    q / ESC = sair

Uso:
    python inspecionar.py <caminho/para/candidatos.csv>
"""

import argparse
import csv
import os

import cv2
import numpy as np

# Importa a visualizacao ja pronta do revisor.
from revisor import imread_unicode, painel_recorte, painel_inteiro, ALTURA_PAINEL

# --- REGRA a inspecionar (a melhor do sweep) e o que mostrar ---
REGRA = {"s": 0.015, "z": 50, "omin": 0.0, "omax": 0.85}
MODO = "fn"   # "fn" = flakes deixados de fora; "fp" = lixo que passou


def carregar(csv_path):
    """Junta candidatos + rotulos, guardando score/size/occ para filtrar/mostrar."""
    base = os.path.dirname(csv_path)
    cand = {r["cand_id"]: r for r in csv.DictReader(open(csv_path, encoding="utf-8"))}
    rot = {r["cand_id"]: r["rotulo"]
           for r in csv.DictReader(open(os.path.join(base, "rotulos.csv"), encoding="utf-8"))}

    itens = []
    for cid, c in cand.items():
        if rot.get(cid) not in ("0", "1"):
            continue
        nums = c["contorno"].split()
        pts = np.array(nums, dtype=np.int32).reshape(-1, 2) if nums else None
        itens.append({
            "cand_id": cid, "imagem": c["imagem"],
            "x": int(c["x"]), "y": int(c["y"]), "w": int(c["w"]), "h": int(c["h"]),
            "contorno": pts,
            "score": float(c["score"]), "size": int(c["size"]), "occ": float(c["max_prob"]),
            "rotulo": rot[cid],
        })
    return itens


def aceito(it):
    R = REGRA
    return (it["score"] >= R["s"] and it["size"] >= R["z"]
            and R["omin"] <= it["occ"] <= R["omax"])


def motivo(it):
    """Por que ficou de fora (para o MODO fn)."""
    R, ms = REGRA, []
    if it["score"] < R["s"]:
        ms.append(f"score {it['score']:.3f}<{R['s']}")
    if it["size"] < R["z"]:
        ms.append(f"size {it['size']}<{R['z']}")
    if it["occ"] > R["omax"]:
        ms.append(f"occ {it['occ']:.3f}>{R['omax']}")
    if it["occ"] < R["omin"]:
        ms.append(f"occ {it['occ']:.3f}<{R['omin']}")
    return ", ".join(ms) if ms else "-"


def main(csv_path):
    itens = carregar(csv_path)
    if MODO == "fn":
        sel = [it for it in itens if it["rotulo"] == "1" and not aceito(it)]
        titulo = "FLAKES deixados de fora (falsos negativos)"
    else:
        sel = [it for it in itens if it["rotulo"] == "0" and aceito(it)]
        titulo = "LIXO que passou (falsos positivos)"
    sel.sort(key=lambda it: it["imagem"])

    pasta_imagens = os.path.dirname(os.path.dirname(csv_path))
    print(f"{titulo}: {len(sel)}   |   regra: {REGRA}")
    print("teclas:  espaco/n=proximo   b=anterior   q=sair")
    if not sel:
        return

    cv2.namedWindow("inspecionar", cv2.WINDOW_NORMAL)
    cache_nome, cache_img = None, None
    i = 0
    while 0 <= i < len(sel):
        it = sel[i]
        if it["imagem"] != cache_nome:
            cache_img = imread_unicode(os.path.join(pasta_imagens, it["imagem"]))
            cache_nome = it["imagem"]
        if cache_img is None:
            i += 1
            continue

        p1 = painel_recorte(cache_img, it, com_contorno=True)
        p2 = painel_recorte(cache_img, it, com_contorno=False)
        p3 = painel_inteiro(cache_img, it)
        sep = np.zeros((ALTURA_PAINEL, 4, 3), dtype=np.uint8)
        corpo = np.hstack([p1, sep, p2, sep, p3])

        txt = (f"{i+1}/{len(sel)}  score={it['score']:.3f} size={it['size']} "
               f"occ={it['occ']:.3f}   motivo: {motivo(it)}")
        barra = np.zeros((34, corpo.shape[1], 3), dtype=np.uint8)
        cv2.putText(barra, txt, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("inspecionar", np.vstack([barra, corpo]))

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("b"):
            i = max(0, i - 1)
        else:  # espaco, n, ou qualquer outra: avanca
            i += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="Caminho para o candidatos.csv")
    args = parser.parse_args()
    main(os.path.abspath(args.csv))
