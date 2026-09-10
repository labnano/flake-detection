"""
Mede a razão de foco em vários recortes de uma imagem grande -- tipicamente
o mosaico de uma varredura já feita.

Para que serve: a razão de medir_foco_relativo só presta como gatilho se
for COMPARÁVEL entre campos de conteúdo diferente. Este script põe essa
promessa à prova sem gastar tempo de microscópio.

Dois desfechos, e os dois são informação:
  - faixa estreita de razões -> a métrica é comparável. Segue o jogo.
  - valores espalhados por uma ordem de grandeza -> ou a métrica responde
    ao CONTEÚDO (e aí não serve de gatilho), ou aquela varredura estava
    mesmo com foco irregular. Os números não desempatam isso sozinhos: é
    para isso que os recortes extremos são salvos em PNG.

Usa memmap e lê só os recortes -- um mosaico de 2 GB nunca é carregado
inteiro em lugar nenhum.

Uso:
    python analisar_foco_razao.py "C:/caminho/mosaico_final.tif"
"""

import json
import os
import sys

import cv2
import numpy as np
import tifffile

from comandos import medir_foco_relativo, medir_textura

# Lado do recorte, em pixels. O ideal é o tamanho de um quadro da câmera:
# é sobre quadros inteiros que a métrica vai rodar durante a varredura.
LADO_RECORTE = 4056

# Recortes por eixo. 8 -> 64 amostras, cerca de um a dois minutos.
GRADE = 8

# Abaixo disto o recorte é substrato limpo ou área preta do mosaico
# (posição da grade sem imagem). Ali a razão mede ruído do sensor, não
# foco -- mesmo critério do TEXTURA_MINIMA do deteccao.py.
TEXTURA_MINIMA_ANALISE = 3.0

# Quantos recortes de cada extremo salvar para conferência visual.
QTD_EXTREMOS = 3


def conferir_resolucao_nativa(caminho):
    caminho_json = os.path.join(os.path.dirname(os.path.abspath(caminho)), "mosaico.json")

    if not os.path.isfile(caminho_json):
        print(
            "[Aviso] mosaico.json não encontrado: não dá para confirmar que este "
            "mosaico está em resolução nativa. Se ele foi montado com "
            "FATOR_REDUCAO != 1, as razões medidas aqui NÃO são comparáveis às "
            "que a varredura vai medir em quadros cheios."
        )
        return

    try:
        with open(caminho_json, encoding="utf-8") as arquivo:
            fator = json.load(arquivo).get("fator_reducao", 1)
    except (json.JSONDecodeError, OSError) as erro:
        print(f"[Aviso] {caminho_json} ilegível ({erro}).")
        return

    if fator != 1:
        raise SystemExit(
            f"ERRO: este mosaico foi montado com FATOR_REDUCAO = {fator}.\n"
            "A razão de foco NÃO é invariante a escala: reduzir a imagem altera "
            "o valor medido mais do que o próprio desfoque que se quer detectar, "
            "então um limiar calibrado aqui ficaria errado.\n"
            "Rode a análise sobre a subpasta mosaico/ (tiles em resolução nativa) "
            "ou remonte o mosaico com FATOR_REDUCAO = 1."
        )


def main(caminho):
    conferir_resolucao_nativa(caminho)

    # mode="r" é deliberado: somente leitura. Sem isso, um engano numa
    # fatia poderia escrever dentro do arquivo original de 2 GB.
    mosaico = tifffile.memmap(caminho, mode="r")
    altura, largura = mosaico.shape[:2]

    print(f"Mosaico: {largura} x {altura} px")

    if altura < LADO_RECORTE or largura < LADO_RECORTE:
        raise SystemExit(f"Imagem menor que LADO_RECORTE ({LADO_RECORTE} px).")

    ys = np.linspace(0, altura - LADO_RECORTE, GRADE).astype(int)
    xs = np.linspace(0, largura - LADO_RECORTE, GRADE).astype(int)

    medidas = []
    descartados = 0

    for y in ys:
        for x in xs:
            # ascontiguousarray força a leitura do disco e devolve um array
            # normal: uma fatia de memmap continua sendo memmap, e o cv2
            # espera memória contígua de verdade.
            recorte = np.ascontiguousarray(
                mosaico[y:y + LADO_RECORTE, x:x + LADO_RECORTE]
            )

            textura = medir_textura(recorte)

            if textura < TEXTURA_MINIMA_ANALISE:
                descartados += 1
                continue

            razao = medir_foco_relativo(recorte)

            if not np.isfinite(razao):
                descartados += 1
                continue

            medidas.append((razao, textura, int(x), int(y)))

    print(
        f"Recortes: {GRADE * GRADE} | com estrutura: {len(medidas)} "
        f"| descartados: {descartados}"
    )

    if len(medidas) < 4:
        raise SystemExit(
            "Amostras demais descartadas por falta de textura. Baixe "
            "TEXTURA_MINIMA_ANALISE ou aumente GRADE."
        )

    medidas.sort()
    razoes = np.array([m[0] for m in medidas])

    print("\n--- RAZÃO POR RECORTE (menor -> maior) ---")
    for razao, textura, x, y in medidas:
        print(f"  razao {razao:.4f} | textura {textura:6.2f} | x={x:6d} y={y:6d}")

    print("\n--- RESUMO ---")
    print(f"mínimo  : {razoes.min():.4f}")
    print(f"mediana : {np.median(razoes):.4f}")
    print(f"máximo  : {razoes.max():.4f}")
    print(f"espalhamento (máx/mín): {razoes.max() / razoes.min():.2f}x")

    # O passo que de fato decide. Se o recorte de menor razão não parecer
    # mais borrado que o de maior, a métrica não está medindo foco -- e aí
    # nenhum limiar conserta.
    pasta = os.path.dirname(os.path.abspath(caminho))

    extremos = (
        ("pior", medidas[:QTD_EXTREMOS]),
        ("melhor", medidas[-QTD_EXTREMOS:]),
    )

    print()
    for rotulo, grupo in extremos:
        for razao, _, x, y in grupo:
            recorte = np.ascontiguousarray(
                mosaico[y:y + LADO_RECORTE, x:x + LADO_RECORTE]
            )
            nome = f"razao_{rotulo}_{razao:.4f}_x{x}_y{y}.png"
            # O mosaico está em RGB; o cv2.imwrite espera BGR.
            gravou = cv2.imwrite(
                os.path.join(pasta, nome),
                cv2.cvtColor(recorte, cv2.COLOR_RGB2BGR),
            )

            if gravou:
                print(f"salvo: {nome}")
            else:
                # imwrite devolve False sem levantar exceção. Acontece em
                # caminho com caractere não-ASCII, e sem esta checagem o
                # arquivo some sem nenhum aviso.
                print(f"[Erro] não foi possível salvar: {os.path.join(pasta, nome)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)

    main(sys.argv[1])