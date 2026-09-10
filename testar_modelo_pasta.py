"""
Re-processa as imagens de uma varredura já salva (a subpasta mosaico/) com o
modelo atual -- incluindo o pós-filtro treinado e a fusão de contornos -- e gera
uma saída no mesmo formato do deteccao.py, pronta pro montar_mosaico.py.

Serve pra ver, no mosaico geral, a performance das métricas novas sem precisar
do microscópio: você aponta pra uma varredura antiga, ele re-detecta com as
regras atuais e monta um mosaico com os flakes contornados.

DOIS MODOS DE CORTE (27/08/2026)
    padrão      -- limiar fixo (parametros.LIMIAR_PROB_FLAKE). Uma passada,
                   rápido. O volume entregue varia entre amostras, porque a
                   probabilidade não significa a mesma coisa em cada uma:
                   medido de 30 a 452 flakes por 1000 tiles nas 6 amostras.
    --orcamento -- você diz quantos flakes por 1000 tiles topa inspecionar e o
                   limiar é calibrado NESTA pasta. Duas passadas (dobra o
                   tempo), volume previsível, e é a forma como os 83% de
                   captura foram medidos. Recomendado: --orcamento 300.

Saída (em TesteDeteccao_<timestamp>/):
    mosaico/ -> TODAS as posições. As com flake ganham o contorno unido
                desenhado; o resto fica original. É o que o montar_mosaico monta.
    flakes/  -> as posições com flake: a imagem marcada (flakes_*) pra
                conferência, e o original (img_pos_*) que o montar_mosaico usa
                pra desenhar a moldura de destaque da posição. Assim o mosaico
                mostra o contorno (forma) E a moldura (mapa grosso da posição).

Rodar: python testar_modelo_pasta.py *pasta mosaico/ da varredura*
       (sem a pasta, abre uma janela pra escolher)
"""

import argparse
import csv
import datetime
import glob
import os
import shutil
import tkinter as tk
from tkinter import filedialog

import time

import cv2
import tifffile

import numpy as np

from caminhos import DIRETORIO_SAIDA_BASE
from modelo import (obter_predictor, filtrar_por_modelo, desenhar_contornos,
                    probabilidades_flake)
from parametros import LIMIAR_PROB_FLAKE


def calibrar_limiar(predictor, caminhos, orcamento):
    """Acha o limiar de probabilidade que entrega `orcamento` flakes por 1000
    tiles NESTA pasta -- em vez de usar o limiar global do parametros.py.

    Por que isso existe: a probabilidade nao significa a mesma coisa em amostras
    diferentes. Exposicao, cor de fundo e densidade de sujeira mudam, e a
    distribuicao de probabilidade escorrega junto. Medido nas 6 amostras de
    19/08: com um limiar unico, o volume entregue varia de 30 a 452 flakes por
    1000 tiles. Escolhendo o limiar DENTRO da amostra, o volume fica onde voce
    pediu -- e foi assim que os 83% de captura foram medidos.

    Isso nao e trapaca: nenhum rotulo e consultado, so as probabilidades. E
    exatamente o que voce faz na bancada -- roda em tudo, ordena, olha os N
    melhores.

    Custo: uma passada extra pela GPU (o dobro do tempo). So vale offline; ao
    vivo, no deteccao.py, nao da para ver a amostra inteira antes de decidir.
    """
    print(f"[1/2] calibrando o limiar para {orcamento:.0f} flakes/1000 tiles...")
    probs = []
    for i, caminho in enumerate(caminhos):
        imagem = cv2.imread(caminho)
        if imagem is None:
            continue
        flakes = predictor.predict(imagem)
        if not flakes:
            continue
        p = probabilidades_flake(imagem, flakes)
        if p is None:                      # sem .joblib: nao ha o que calibrar
            return None
        probs.extend(p.tolist())
        if (i + 1) % 200 == 0:
            print(f"      {i+1}/{len(caminhos)} tiles")

    alvo = int(round(orcamento * len(caminhos) / 1000))
    if not probs or alvo <= 0:
        return 1.1                          # nada cabe: rejeita tudo
    ordenadas = np.sort(np.array(probs))[::-1]
    if alvo >= len(ordenadas):
        return 0.0                          # cabe tudo
    limiar = float(ordenadas[alvo])
    print(f"      {len(probs)} candidatos | limiar = {limiar:.4f} "
          f"(global seria {LIMIAR_PROB_FLAKE})")
    return limiar


def main(pasta, saida, orcamento=None):

    predictor = obter_predictor()

    caminhos = glob.glob(os.path.join(pasta, "**", "*.tif"), recursive=True)
    if not caminhos:
        print(f"Nenhuma imagem .tif encontrada em: {pasta}")
        return

    limiar = LIMIAR_PROB_FLAKE
    if orcamento is not None:
        calibrado = calibrar_limiar(predictor, caminhos, orcamento)
        if calibrado is not None:
            limiar = calibrado
        print("[2/2] detectando...")

    pasta_mosaico = os.path.join(saida, "mosaico")
    pasta_flakes = os.path.join(saida, "flakes")
    os.makedirs(pasta_mosaico, exist_ok=True)
    os.makedirs(pasta_flakes, exist_ok=True)

    # O montar_mosaico usa o varredura.json (passo real, sobreposição, grade)
    # pra encaixar os tiles sem duplicar a faixa compartilhada. Ele fica na
    # RAIZ da varredura (um nível acima da pasta mosaico/), então copiamos.
    json_orig = os.path.join(os.path.dirname(pasta.rstrip("\\/")), "varredura.json")
    if os.path.isfile(json_orig):
        shutil.copy(json_orig, os.path.join(saida, "varredura.json"))
    else:
        print("[aviso] varredura.json não achado -- o mosaico assumirá tiles adjacentes.")
    inicio = time.perf_counter()
    com_flake = 0
    # Uma linha por flake aceito, para o relatorio ordenado do fim.
    achados = []
    por_tile = []          # (melhor prob do tile, nome) -- vira o ranking
    for caminho in caminhos:
        # cv2.imread devolve BGR uint8 -- que é o que o modelo espera.
        imagem_bgr = cv2.imread(caminho)
        if imagem_bgr is None:
            print(f"[Erro] Não consegui ler: {caminho}")
            continue

        nome = os.path.basename(caminho)
        flakes = predictor.predict(imagem_bgr)
        # Modelo treinado (contraste BGR + frac_substrato): separa flake fino de
        # cola E de bulk, o que o teto de contraste verde não conseguia fazer.
        flakes = filtrar_por_modelo(imagem_bgr, flakes, limiar)

        if flakes:
            com_flake += 1
            marcada_bgr = desenhar_contornos(imagem_bgr, flakes)

            melhor = max((getattr(f, "prob_flake", 1.0) for f in flakes),
                         default=1.0)
            for f in flakes:
                achados.append((float(getattr(f, "prob_flake", 1.0)), nome,
                                int(f.size), int(f.center[0]), int(f.center[1])))
            # Guarda para o ranking do fim: o rank só existe depois que TODOS os
            # tiles foram vistos, então o arquivo nasce com o nome simples e é
            # renomeado no final.
            por_tile.append((melhor, nome))

            # tifffile grava o array como está; a varredura é RGB, então
            # convertemos BGR->RGB pra manter as cores certas no mosaico.
            tifffile.imwrite(os.path.join(pasta_mosaico, nome), marcada_bgr[:, :, ::-1])
            tifffile.imwrite(os.path.join(pasta_flakes, "flakes_" + nome),
                             marcada_bgr[:, :, ::-1])
            # Original em flakes/ com o nome img_pos_*: é o que o montar_mosaico
            # procura pra desenhar a moldura de destaque da posição no mosaico.
            # (Pra voltar a "só contorno", basta remover esta linha.)
            tifffile.imwrite(os.path.join(pasta_flakes, nome), imagem_bgr[:, :, ::-1])
            print(f"{nome}: {len(flakes)} flake(s)")
        else:
            # Sem flake: a posição vai pro mosaico como original, pra grade fechar.
            tifffile.imwrite(os.path.join(pasta_mosaico, nome), imagem_bgr[:, :, ::-1])
            print(f"{nome}: -")
    fim = time.perf_counter()

    # RANKING: renomeia os arquivos de conferência com r### na ordem de
    # inspeção. O rank só é conhecido depois de ver todos os tiles, por isso a
    # renomeação vem aqui e não no laço.
    #
    # Por que rank e não a probabilidade no nome: o explorador ordena por nome
    # em ordem CRESCENTE, então um prefixo de probabilidade traria os piores
    # primeiro. Com r001, r002, ..., a ordem natural já é a ordem em que vale a
    # pena olhar. A probabilidade vai junto, depois, para você saber onde está.
    por_tile.sort(reverse=True)
    for rank, (prob, nome) in enumerate(por_tile, 1):
        antigo = os.path.join(pasta_flakes, "flakes_" + nome)
        novo = os.path.join(pasta_flakes,
                            f"r{rank:03d}_p{int(round(1000*prob)):03d}_" + nome)
        if os.path.exists(antigo):
            os.replace(antigo, novo)

    # Relatório ordenado: a lista de inspeção, do mais provável ao menos.
    if achados:
        achados.sort(reverse=True)
        with open(os.path.join(saida, "flakes.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["prob", "imagem", "size_px", "centro_x", "centro_y"])
            for prob, img, size, cx, cy in achados:
                w.writerow([f"{prob:.4f}", img, size, cx, cy])

        print(f"\n{len(achados)} flake(s) aceitos. Os 10 mais prováveis:")
        for prob, img, size, _, _ in achados[:10]:
            print(f"  {prob:.3f}  {img:<28} {size:>6} px")
        # A precisão medida no ponto de operação é ~62%: o fim da lista é onde
        # moram os falsos positivos. Ordenar por probabilidade transforma
        # "tem falso positivo" em "os falsos positivos vêm por último".
        print(f"\nlista completa ordenada: {os.path.join(saida, 'flakes.csv')}")
        print(f"\nlista de inspeção: {len(por_tile)} posições, renomeadas")
        print("  r001_p829_img_pos_...  ->  rank 001, probabilidade 0,829")
        print("Ordene flakes/ por nome e olhe de cima para baixo; a precisão")
        print("medida é ~62%, então o lixo se concentra no fim da lista.")

    print(f"\n{com_flake} posição(ões) com flake de {len(caminhos)} imagens.")
    print(f"Saída em: {saida}")
    print("Para montar o mosaico:")
    print(f'    python montar_mosaico.py "{saida}"')
    print(f" tempo total de detecção = {fim - inicio}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pasta", nargs="?", default=None,
                        help="Pasta com as imagens .tif (se omitida, abre uma janela)")
    parser.add_argument("--saida", default=None,
                        help="Pasta de saída (default: <BASE>/TesteDeteccao_<timestamp>)")
    parser.add_argument("--orcamento", type=float, default=None,
                        help="Flakes por 1000 tiles que você topa inspecionar. "
                             "Com esta opção o limiar é calibrado NESTA pasta "
                             "(duas passadas, dobra o tempo) em vez de usar o "
                             "LIMIAR_PROB_FLAKE global. Recomendado: 300.")
    args = parser.parse_args()

    pasta = args.pasta
    if pasta is None:
        raiz = tk.Tk()
        raiz.withdraw()  # esconde a janela-raiz vazia do tkinter
        pasta = filedialog.askdirectory(
            title="Escolha a pasta com as imagens .tif",
            initialdir=DIRETORIO_SAIDA_BASE,
        )
        raiz.destroy()
        if not pasta:
            print("Nenhuma pasta escolhida.")
            raise SystemExit

    saida = args.saida
    if saida is None:
        ts = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        saida = os.path.join(DIRETORIO_SAIDA_BASE, "TesteDeteccao_" + ts)

    main(pasta, saida, args.orcamento)
