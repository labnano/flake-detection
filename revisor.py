"""
revisor.py  --  revisao de rotulos com 3 paineis (dúvidas + outliers).

Para que serve
--------------
Depois de anotar tudo (anotador.py), este script reabre SO os casos que valem
uma segunda olhada, mostrando 3 vistas lado a lado:
    [1] recorte COM o contorno da forma detectada
    [2] o mesmo recorte SEM contorno (o flake limpo, para comparar com o substrato)
    [3] a imagem INTEIRA da varredura, com o candidato destacado (contexto)

Tres grupos entram na fila:
    - duvida      : rotulo '?'
    - lixo_alto   : marcado LIXO(0) mas com score >= LIMIAR_LIXO_ALTO
    - flake_baixo : marcado FLAKE(1) mas com score <  LIMIAR_FLAKE_BAIXO

Anti-vies (importante)
----------------------
O objetivo do experimento e AVALIAR o score; se a gente "corrige" rotulos para
concordar com o score, contamina o gabarito com a propria variavel avaliada
(circularidade). Por isso o score fica ESCONDIDO por padrao -- voce julga a
imagem pelo que ela e. So mude um rotulo se a IMAGEM mostrar que voce clicou
errado, nunca porque o score e alto/baixo. (Tecla 's' revela o score, mas use
com consciencia.)

Uso:
    python revisor.py <caminho/para/candidatos.csv>
"""

import argparse
import csv
import os

import cv2
import numpy as np

# --- QUAIS OUTLIERS ENTRAM (aperte/afrouxe aqui) ---
LIMIAR_LIXO_ALTO = 0.30    # lixo com score >= isso entra na fila
LIMIAR_FLAKE_BAIXO = 0.05  # flake com score < isso entra na fila

# --- VISUAL ---
ALTURA_PAINEL = 480        # altura comum dos 3 paineis, em pixels
MARGEM_CROP = 40           # substrato ao redor do flake nos recortes
ESPESSURA = 1              # contorno fino
COR_CONTORNO = (0, 0, 255)   # BGR vermelho (recorte 1)
COR_MARCA = (0, 255, 0)      # BGR verde (destaque na imagem inteira)

NOME_ROTULO = {"1": "FLAKE", "0": "LIXO", "?": "DUVIDA", "": "-"}


def imread_unicode(caminho):
    dados = np.fromfile(caminho, dtype=np.uint8)
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def carregar_candidatos(csv_path):
    """cand_id -> geometria (imagem, bbox, contorno). So o que o revisor desenha."""
    d = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nums = row["contorno"].split()
            pts = np.array(nums, dtype=np.int32).reshape(-1, 2) if nums else None
            d[row["cand_id"]] = {
                "imagem": row["imagem"],
                "x": int(row["x"]), "y": int(row["y"]),
                "w": int(row["w"]), "h": int(row["h"]),
                "contorno": pts,
            }
    return d


def carregar_rotulos(path):
    """Le rotulos.csv. Aceita o formato do anotador (3 colunas) ou o ja revisado
    (com rotulo_original e revisado)."""
    reg = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reg[row["cand_id"]] = {
                "cand_id": row["cand_id"],
                "rotulo": row["rotulo"],
                "score": float(row["score"]),
                "rotulo_original": row.get("rotulo_original", ""),
                "revisado": row.get("revisado", "0"),
            }
    return reg


def salvar_rotulos(path, reg):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cand_id", "rotulo", "score", "rotulo_original", "revisado"])
        for r in reg.values():
            w.writerow([r["cand_id"], r["rotulo"], f"{r['score']:.6f}",
                        r["rotulo_original"], r["revisado"]])


def construir_fila(cands, reg):
    """Monta a lista de candidatos a revisar (os 3 grupos, menos os ja revisados)."""
    fila = []
    for cid, r in reg.items():
        if r["revisado"] == "1":
            continue
        rot, sc = r["rotulo"], r["score"]
        if rot == "?":
            grupo = "duvida"
        elif rot == "0" and sc >= LIMIAR_LIXO_ALTO:
            grupo = "lixo_alto"
        elif rot == "1" and sc < LIMIAR_FLAKE_BAIXO:
            grupo = "flake_baixo"
        else:
            continue
        if cid not in cands:
            continue
        item = dict(cands[cid])
        item.update({"cand_id": cid, "rotulo": rot, "score": sc, "grupo": grupo})
        fila.append(item)
    fila.sort(key=lambda it: it["imagem"])  # agrupa por imagem -> reabre menos
    return fila


def _escala_para_altura(img, altura, ampliar_nn=True):
    """Redimensiona para uma altura fixa, devolvendo (img, fator)."""
    h = img.shape[0]
    esc = altura / h
    interp = cv2.INTER_NEAREST if (esc > 1 and ampliar_nn) else cv2.INTER_AREA
    larg = max(1, int(round(img.shape[1] * esc)))
    return cv2.resize(img, (larg, altura), interpolation=interp), esc


def painel_recorte(imagem, cand, com_contorno):
    """Recorte do candidato + margem, na altura comum; contorno opcional."""
    H, W = imagem.shape[:2]
    x, y, w, h = cand["x"], cand["y"], cand["w"], cand["h"]
    x0, y0 = max(x - MARGEM_CROP, 0), max(y - MARGEM_CROP, 0)
    x1, y1 = min(x + w + MARGEM_CROP, W), min(y + h + MARGEM_CROP, H)
    crop = imagem[y0:y1, x0:x1].copy()
    crop, esc = _escala_para_altura(crop, ALTURA_PAINEL)
    if com_contorno and cand["contorno"] is not None and len(cand["contorno"]) >= 2:
        p = ((cand["contorno"] - np.array([x0, y0])) * esc).astype(np.int32)
        cv2.polylines(crop, [p], isClosed=True, color=COR_CONTORNO, thickness=ESPESSURA)
    return crop


def painel_inteiro(imagem, cand):
    """A imagem toda, reduzida, com o candidato destacado (retangulo + mira)."""
    img, esc = _escala_para_altura(imagem, ALTURA_PAINEL, ampliar_nn=False)
    x, y, w, h = cand["x"], cand["y"], cand["w"], cand["h"]
    rx0, ry0 = int(x * esc), int(y * esc)
    rx1, ry1 = int((x + w) * esc), int((y + h) * esc)
    cv2.rectangle(img, (rx0 - 6, ry0 - 6), (rx1 + 6, ry1 + 6), COR_MARCA, 2)
    cx, cy = (rx0 + rx1) // 2, (ry0 + ry1) // 2
    cv2.line(img, (cx, 0), (cx, img.shape[0]), COR_MARCA, 1)
    cv2.line(img, (0, cy), (img.shape[1], cy), COR_MARCA, 1)
    return img


def compor(imagem, cand, texto_barra):
    p1 = painel_recorte(imagem, cand, com_contorno=True)
    p2 = painel_recorte(imagem, cand, com_contorno=False)
    p3 = painel_inteiro(imagem, cand)
    sep = np.zeros((ALTURA_PAINEL, 4, 3), dtype=np.uint8)
    corpo = np.hstack([p1, sep, p2, sep, p3])

    barra = np.zeros((34, corpo.shape[1], 3), dtype=np.uint8)
    cv2.putText(barra, texto_barra, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([barra, corpo])


def main(csv_path):
    pasta_imagens = os.path.dirname(os.path.dirname(csv_path))
    rotulos_path = os.path.join(os.path.dirname(csv_path), "rotulos.csv")

    cands = carregar_candidatos(csv_path)
    reg = carregar_rotulos(rotulos_path)
    fila = construir_fila(cands, reg)

    # Contagem por grupo, so para voce saber o tamanho da tarefa.
    from collections import Counter
    cont = Counter(it["grupo"] for it in fila)
    print(f"a revisar: {len(fila)}  "
          f"(duvida={cont.get('duvida',0)}  "
          f"lixo_alto={cont.get('lixo_alto',0)}  flake_baixo={cont.get('flake_baixo',0)})")
    print("teclas:  espaco=manter  1=flake  0=lixo  d=duvida  s=mostra/esconde score  u=undo  q=sair")
    if not fila:
        print("nada a revisar com os limiares atuais.")
        return

    cv2.namedWindow("revisor", cv2.WINDOW_NORMAL)
    cache_nome, cache_img = None, None
    mostrar_score = False
    snapshots = {}  # cand_id -> estado do registro antes da revisao (para undo)

    i = 0
    while i < len(fila):
        item = fila[i]
        cid = item["cand_id"]

        if item["imagem"] != cache_nome:
            cache_img = imread_unicode(os.path.join(pasta_imagens, item["imagem"]))
            cache_nome = item["imagem"]
        if cache_img is None:
            print(f"[erro] nao li {item['imagem']}, pulando")
            i += 1
            continue

        atual = NOME_ROTULO.get(reg[cid]["rotulo"], "?")
        texto = (f"{i+1}/{len(fila)}  atual: {atual}   "
                 f"espaco=manter 1=flake 0=lixo d=duvida s=score u=undo q=sair")
        if mostrar_score:
            texto += f"   [score={item['score']:.3f} grupo={item['grupo']}]"

        cv2.imshow("revisor", compor(cache_img, item, texto))
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            mostrar_score = not mostrar_score
            continue  # re-exibe o MESMO item, sem avancar
        elif key == ord("u"):
            i = max(0, i - 1)
            cid_ant = fila[i]["cand_id"]
            if cid_ant in snapshots:
                reg[cid_ant] = dict(snapshots.pop(cid_ant))
                salvar_rotulos(rotulos_path, reg)
            continue
        elif key in (ord(" "), ord("k"), 13, ord("1"), ord("0"), ord("d")):
            # guarda o estado original UMA vez, para o undo poder reverter
            if cid not in snapshots:
                snapshots[cid] = dict(reg[cid])
            # 'manter' nao muda o rotulo; 1/0/d trocam
            if key in (ord("1"), ord("0"), ord("d")):
                novo = {ord("1"): "1", ord("0"): "0", ord("d"): "?"}[key]
                if reg[cid]["rotulo_original"] == "":
                    reg[cid]["rotulo_original"] = reg[cid]["rotulo"]
                reg[cid]["rotulo"] = novo
            reg[cid]["revisado"] = "1"
            salvar_rotulos(rotulos_path, reg)
            i += 1
        # outras teclas: ignora, re-exibe

    cv2.destroyAllWindows()
    revisados = sum(1 for r in reg.values() if r["revisado"] == "1")
    mudados = sum(1 for r in reg.values()
                  if r["rotulo_original"] not in ("", r["rotulo"]))
    print(f"\nrevisados: {revisados}  |  rotulos mudados: {mudados}")
    print(f"salvo em: {rotulos_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="Caminho para o candidatos.csv")
    args = parser.parse_args()
    main(os.path.abspath(args.csv))
