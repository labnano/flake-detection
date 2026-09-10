"""
anotador.py  --  PASSO 2 de 3 (versao definitiva).

O que faz
---------
Le o candidatos.csv do passo 1 e mostra TODOS os candidatos, um de cada vez,
para voce rotular no teclado. Recorta da imagem original na hora e desenha o
CONTORNO da forma que o modelo detectou (nao so uma caixa), fino, para voce
comparar o flake com o substrato ao redor.

    1 = e flake de verdade
    0 = e lixo / falso positivo
    d = duvida (fica marcado com '?', a gente decide depois)
    u = desfazer o ultimo
    q = salvar e sair   (ESC tambem)

Sem amostragem: como o extrator ja filtra no piso 0.015, o CSV so tem os
candidatos que valem a pena olhar. Voce anota todos. Os rotulos vao para
rotulos.csv (ao lado do candidatos.csv), salvos a CADA clique -- da pra parar
no meio e retomar depois, que ele pula o que ja foi feito.

Uso:
    python anotador.py <caminho/para/candidatos.csv>
    (a pasta das imagens e inferida como .../<mosaico>, dois niveis acima do
     CSV; use --imagens para apontar outra.)
"""

import argparse
import csv
import os

import cv2
import numpy as np

# --- VISUAL (mexa aqui para ajustar o recorte e o destaque) ---
MARGEM_CROP = 40      # pixels de substrato mostrados ao redor do flake
ESPESSURA = 1         # espessura do contorno -- fino, para nao "engolir" a borda
COR = (0, 0, 255)     # BGR: vermelho
ALVO_MIN = 450        # se o recorte for menor que isso, amplia (ver flakes pequenos)
ALVO_MAX = 900        # se for maior, reduz (caber na tela)


def imread_unicode(caminho):
    """Le imagem mesmo com acento no caminho (mesmo truque do passo 1)."""
    dados = np.fromfile(caminho, dtype=np.uint8)
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def carregar_candidatos(csv_path):
    """Le o candidatos.csv inteiro; converte bbox e contorno para numeros."""
    out = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nums = row["contorno"].split()
            # Reconstroi o poligono Nx2 a partir de "x1 y1 x2 y2 ..." (coords globais).
            pts = np.array(nums, dtype=np.int32).reshape(-1, 2) if nums else None
            out.append({
                "cand_id": row["cand_id"],
                "imagem": row["imagem"],
                "score": float(row["score"]),
                "x": int(row["x"]), "y": int(row["y"]),
                "w": int(row["w"]), "h": int(row["h"]),
                "contorno": pts,
            })
    return out


def carregar_rotulos(path):
    """Le rotulos.csv se existir -> dict cand_id -> registro (para retomar)."""
    registros = {}
    if not os.path.exists(path):
        return registros
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            registros[row["cand_id"]] = {
                "cand_id": row["cand_id"],
                "rotulo": row["rotulo"],
                "score": float(row["score"]),
            }
    return registros


def salvar_rotulos(path, registros):
    """Reescreve rotulos.csv inteiro (poucos milhares de linhas, instantaneo)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cand_id", "rotulo", "score"])
        for r in registros.values():
            w.writerow([r["cand_id"], r["rotulo"], f"{r['score']:.6f}"])


def preparar_crop(imagem, cand):
    """Recorta o candidato + margem, amplia/reduz para caber, e desenha o
    contorno FINO da forma detectada (transladado e escalado para o recorte)."""
    H, W = imagem.shape[:2]
    x, y, w, h = cand["x"], cand["y"], cand["w"], cand["h"]

    x0, y0 = max(x - MARGEM_CROP, 0), max(y - MARGEM_CROP, 0)
    x1, y1 = min(x + w + MARGEM_CROP, W), min(y + h + MARGEM_CROP, H)
    crop = imagem[y0:y1, x0:x1].copy()

    # Escala: amplia recorte pequeno (pixel real, sem suavizar) ou reduz o grande.
    lado = max(crop.shape[0], crop.shape[1])
    if lado < ALVO_MIN:
        escala = float(ALVO_MIN // lado + 1)          # fator inteiro
        crop = cv2.resize(crop, None, fx=escala, fy=escala,
                          interpolation=cv2.INTER_NEAREST)
    elif lado > ALVO_MAX:
        escala = ALVO_MAX / lado
        crop = cv2.resize(crop, None, fx=escala, fy=escala,
                          interpolation=cv2.INTER_AREA)
    else:
        escala = 1.0

    # Contorno: leva do sistema da imagem (global) para o do recorte e escala.
    pts = cand["contorno"]
    if pts is not None and len(pts) >= 2:
        p = ((pts - np.array([x0, y0])) * escala).astype(np.int32)
        cv2.polylines(crop, [p], isClosed=True, color=COR, thickness=ESPESSURA)
    return crop


def main(csv_path, pasta_imagens):
    cands = carregar_candidatos(csv_path)
    cands.sort(key=lambda c: c["imagem"])  # agrupa por imagem -> reabre menos

    rotulos_path = os.path.join(os.path.dirname(csv_path), "rotulos.csv")
    registros = carregar_rotulos(rotulos_path)

    print(f"imagens em: {pasta_imagens}")
    print(f"candidatos: {len(cands)} | ja rotulados: {len(registros)}")
    print("teclas:  1=flake  0=lixo  d=duvida  u=desfazer  q=salvar e sair")

    cv2.namedWindow("anotador", cv2.WINDOW_NORMAL)
    cache_nome, cache_img = None, None
    historico = []  # cand_ids rotulados NESTA sessao, para o 'u' (undo)

    i = 0
    while i < len(cands):
        cand = cands[i]
        if cand["cand_id"] in registros:   # ja rotulado (sessao anterior): pula
            i += 1
            continue

        if cand["imagem"] != cache_nome:   # troca de imagem -> recarrega
            cache_img = imread_unicode(os.path.join(pasta_imagens, cand["imagem"]))
            cache_nome = cand["imagem"]
        if cache_img is None:
            print(f"[erro] nao li {cand['imagem']}, pulando")
            i += 1
            continue

        crop = preparar_crop(cache_img, cand)
        titulo = (f"{len(registros) + 1}/{len(cands)}  score={cand['score']:.3f}  "
                  f"| 1=flake 0=lixo d=duvida u=undo q=sair")
        cv2.imshow("anotador", crop)
        cv2.setWindowTitle("anotador", titulo)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):          # q ou ESC
            break
        elif key == ord("u"):              # desfazer
            if historico:
                ultimo = historico.pop()
                registros.pop(ultimo, None)
                salvar_rotulos(rotulos_path, registros)
                for j, c in enumerate(cands):
                    if c["cand_id"] == ultimo:
                        i = j
                        break
            continue
        elif key in (ord("1"), ord("0"), ord("d")):
            rot = {ord("1"): "1", ord("0"): "0", ord("d"): "?"}[key]
            registros[cand["cand_id"]] = {
                "cand_id": cand["cand_id"], "rotulo": rot, "score": cand["score"],
            }
            historico.append(cand["cand_id"])
            salvar_rotulos(rotulos_path, registros)  # salva a cada clique
            i += 1
        # qualquer outra tecla: nao faz nada, reexibe o mesmo candidato

    cv2.destroyAllWindows()
    print(f"\nsalvo em: {rotulos_path}")
    print(f"total rotulado: {len(registros)} de {len(cands)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="Caminho para o candidatos.csv (passo 1)")
    parser.add_argument("--imagens", default=None,
                        help="Pasta das imagens .tif (default: dois niveis acima do CSV)")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    # O candidatos.csv fica em <imagens>/avaliacao_parametros/candidatos.csv,
    # entao a pasta das imagens e o diretorio-avo do CSV.
    pasta_imagens = args.imagens or os.path.dirname(os.path.dirname(csv_path))
    main(csv_path, pasta_imagens)
