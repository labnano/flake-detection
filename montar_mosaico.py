"""
Monta o mosaico final a partir das imagens que uma varredura já deixou
salvas em disco (deteccao.py ou varredura_mosaico.py).

Por que é um script separado da varredura: montar o mosaico é a etapa mais
cara em memória do pipeline. Rodando junto com a varredura, uma falha aqui
(memória estourada, disco cheio) jogaria fora horas de microscópio. Como os
tiles já estão em disco, separar torna a montagem repetível e sem risco --
dá pra tentar de novo, mudar o FATOR_REDUCAO, ou montar só depois.

Uso:
    python montar_mosaico.py "D:\\Users\\PedroGuedes\\Deteccao_05-08-2026_14-30-00"

Sem argumento, o script pergunta o caminho. Se a pasta indicada tiver uma
subpasta "mosaico" (caso do deteccao.py), ele usa ela automaticamente; se
os .tif estiverem soltos na própria pasta (caso do varredura_mosaico.py),
usa a pasta mesmo.

Se existir uma subpasta "flakes" ao lado, as posições que deram positivo
ganham uma moldura colorida no mosaico -- assim ele também serve de mapa
de onde procurar. Sem essa pasta, o mosaico sai sem destaques.
"""

import json
import os
import re
import sys

import cv2
import numpy as np
import tifffile

# Reduz cada imagem antes de encaixar. 1 = resolução total.
# Um mosaico de detecção (FOV_DETECCAO, tiles pequenos) pode passar
# facilmente de dezenas de milhares de pixels de lado -- um arquivo que
# nenhum visualizador comum abre. 4 ou 8 gera algo bem mais leve, ainda
# suficiente pra enxergar onde as coisas estão na amostra.
FATOR_REDUCAO = 1

# --- DESTAQUE DAS POSIÇÕES COM FLAKE ---
# Cor da moldura, em RGB 0-255 (as imagens saem em RGB da capturar_imagem).
COR_DESTAQUE = (255, 0, 0)

# Espessura da moldura como FRAÇÃO do lado do tile já reduzido.
# Definir em fração da saída, e não em pixels da imagem original, é o que
# garante que a moldura continue visível em qualquer FATOR_REDUCAO -- e,
# mais importante, quando você abre o mosaico inteiro num visualizador,
# que reduz tudo de novo. Uma moldura de N pixels da imagem original
# viraria sub-pixel nessa escala e simplesmente sumiria.
ESPESSURA_DESTAQUE = 0.04
ESPESSURA_MINIMA_PX = 2

# Padrão dos nomes gravados pela varredura: img_pos_X{coluna}_Y{linha}.tif
# Os \d+ capturam os índices i (eixo X) e j (eixo Y) do laço da varredura.
PADRAO_NOME = re.compile(r"^img_pos_X(\d+)_Y(\d+)\.tif$")


def descobrir_pasta_das_imagens(pasta):
    """Aceita tanto a pasta raiz da varredura quanto a subpasta que de fato
    contém os .tif, pra funcionar com o layout dos dois scripts de varredura."""

    subpasta = os.path.join(pasta, "mosaico")

    if os.path.isdir(subpasta):
        return subpasta

    return pasta


def mapear_imagens(pasta):
    """Varre a pasta e devolve {(i, j): caminho} para cada arquivo cujo nome
    bate com o padrão da varredura.

    Ler a grade a partir dos próprios nomes de arquivo -- em vez de receber
    GRID_X e GRID_Y de fora -- é o que deixa este script rodar sozinho, sem
    precisar saber nada sobre a varredura que gerou as imagens."""

    posicoes = {}

    for nome in os.listdir(pasta):
        casamento = PADRAO_NOME.match(nome)

        if casamento is None:
            continue

        i = int(casamento.group(1))
        j = int(casamento.group(2))
        posicoes[(i, j)] = os.path.join(pasta, nome)

    return posicoes


def ler_metadados(pasta):
    """Lê o varredura.json que a varredura gravou, se existir.

    Ele diz qual passo o estágio realmente usou. Como o passo pode ser
    menor que o campo (sobreposição entre tiles), sem essa informação o
    montador encaixaria os tiles como se fossem adjacentes e a faixa
    compartilhada apareceria DUAS vezes no mosaico -- conteúdo repetido e
    escala esticada.

    Devolve None se não houver arquivo, o que mantém funcionando as pastas
    de varreduras antigas, feitas antes da sobreposição existir."""

    caminho = os.path.join(pasta, "varredura.json")

    if not os.path.isfile(caminho):
        return None

    try:
        with open(caminho, encoding="utf-8") as arquivo:
            return json.load(arquivo)
    except (json.JSONDecodeError, OSError) as erro:
        print(f"[Aviso] {caminho} ilegível ({erro}); assumindo tiles adjacentes.")
        return None


def cor_do_destaque(dtype, ndim):
    """Converte COR_DESTAQUE para a escala e o formato do tipo de pixel das
    imagens da varredura.

    Sem essa conversão, um vermelho (255, 0, 0) desenhado numa imagem
    uint16 sairia praticamente preto: 255 é ~0,4% do máximo (65535). Em
    imagem monocromática a cor perde sentido, então o destaque vira o
    valor máximo (branco)."""

    maximo = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0

    if ndim == 2:
        return maximo

    return tuple(c / 255 * maximo for c in COR_DESTAQUE)


def desenhar_destaque(img, cor):
    """Desenha uma moldura POR DENTRO das bordas do tile.

    'Por dentro' é essencial: crescer a imagem quebraria o encaixe na grade
    do mosaico, já que todo tile precisa manter exatamente o mesmo tamanho.
    A moldura cobre uma faixa fina da própria imagem -- o que se perde é a
    borda do campo de visão, justamente a região que costuma sobrepor o
    tile vizinho."""

    altura, largura = img.shape[:2]

    espessura = max(
        ESPESSURA_MINIMA_PX,
        int(round(min(altura, largura) * ESPESSURA_DESTAQUE)),
    )

    img[:espessura, :] = cor    # topo
    img[-espessura:, :] = cor   # base
    img[:, :espessura] = cor    # esquerda
    img[:, -espessura:] = cor   # direita

    return img


def main():
    if len(sys.argv) > 1:
        pasta = sys.argv[1]
    else:
        # .strip('"') porque copiar caminho no Windows costuma trazer aspas junto.
        pasta = input("Caminho da pasta da varredura: ").strip().strip('"')

    if not os.path.isdir(pasta):
        print(f"[Erro] Pasta não encontrada: {pasta}")
        return

    pasta_imagens = descobrir_pasta_das_imagens(pasta)
    print(f"Lendo imagens de: {pasta_imagens}")

    posicoes = mapear_imagens(pasta_imagens)

    if not posicoes:
        print("[Erro] Nenhum arquivo no formato img_pos_X<i>_Y<j>.tif encontrado.")
        return

    # A grade é o maior índice visto + 1 (os índices começam em 0).
    grid_x = max(i for i, _ in posicoes) + 1
    grid_y = max(j for _, j in posicoes) + 1
    esperadas = grid_x * grid_y

    print(f"Grade detectada: {grid_x} x {grid_y} ({esperadas} posições)")
    print(f"Imagens encontradas: {len(posicoes)}")

    if len(posicoes) < esperadas:
        print(
            f"[Aviso] {esperadas - len(posicoes)} posição(ões) sem arquivo -- "
            "ficarão pretas no mosaico."
        )

    # Quais posições deram positivo, pra destacar no mosaico. A própria
    # existência do arquivo em flakes/ já é a informação -- não precisa ler
    # o CSV. Reusa o mapear_imagens porque flakes/ guarda os originais com
    # o mesmo padrão de nome (os flakes_pos_*.tif marcados não casam com o
    # padrão e são ignorados de graça).
    pasta_flakes = os.path.join(pasta, "flakes")

    if os.path.isdir(pasta_flakes):
        posicoes_com_flake = set(mapear_imagens(pasta_flakes))
        print(f"Posições com flake (serão destacadas): {len(posicoes_com_flake)}")
    else:
        posicoes_com_flake = set()
        print("(sem subpasta flakes/ -- mosaico sairá sem destaques)")

    # Lê uma imagem só pra descobrir dimensões e tipo dos pixels, sem
    # carregar a varredura inteira.
    amostra = tifffile.imread(next(iter(posicoes.values())))
    altura_tile, largura_tile = amostra.shape[:2]
    canais = amostra.shape[2] if amostra.ndim == 3 else None

    # Quanto de cada tile é conteúdo NOVO (não compartilhado com o vizinho).
    # Sem metadados, assume-se 100% -- tiles adjacentes, comportamento antigo.
    metadados = ler_metadados(pasta)

    if metadados:
        fracao_x = metadados["passo_x"] / metadados["fov_x"]
        fracao_y = metadados["passo_y"] / metadados["fov_y"]
        print(
            f"Varredura feita com {metadados['sobreposicao']:.0%} de sobreposição: "
            f"usando {fracao_x:.0%} x {fracao_y:.0%} de cada tile."
        )
    else:
        fracao_x = fracao_y = 1.0
        print("(sem varredura.json -- assumindo tiles adjacentes, sem sobreposição)")

    # Recorte em pixels da imagem ORIGINAL. Fica-se com o canto onde o tile
    # começa: a região [i*passo, (i+1)*passo] é a parte que pertence a este
    # tile, e o resto já é coberto pelo vizinho seguinte.
    recorte_x = max(1, round(largura_tile * fracao_x))
    recorte_y = max(1, round(altura_tile * fracao_y))

    largura = max(1, recorte_x // FATOR_REDUCAO)
    altura = max(1, recorte_y // FATOR_REDUCAO)

    forma_final = (grid_y * altura, grid_x * largura)

    if canais:
        forma_final += (canais,)

    bytes_totais = int(np.prod(forma_final)) * amostra.dtype.itemsize

    print(
        f"Tile: {largura_tile} x {altura_tile} px -> recortado para "
        f"{recorte_x} x {recorte_y} -> encaixado como {largura} x {altura} | "
        f"dtype: {amostra.dtype}"
    )
    print(
        f"Mosaico final: {forma_final[1]} x {forma_final[0]} px "
        f"(~{bytes_totais / 1e9:.2f} GB)"
    )

    caminho_saida = os.path.join(pasta, "mosaico_final.tif")

    # tifffile.memmap cria o arquivo já no tamanho final e devolve um array
    # "mapeado em disco": escrever nele grava direto no arquivo, sem nunca
    # segurar o mosaico inteiro na RAM. É o que permite montar mosaicos
    # maiores que a memória da máquina -- o np.hstack/vstack da versão
    # antiga precisava do mosaico todo em memória de uma vez.
    # O TIFF clássico tem limite de 4 GB; acima disso é obrigatório BigTIFF.
    mosaico = tifffile.memmap(
        caminho_saida,
        shape=forma_final,
        dtype=amostra.dtype,
        bigtiff=bytes_totais > 3.5e9,
    )

    total = len(posicoes)
    cor_destaque = cor_do_destaque(amostra.dtype, amostra.ndim)

    for contador, ((i, j), caminho) in enumerate(sorted(posicoes.items()), start=1):
        img = tifffile.imread(caminho)

        if img.shape[:2] != (altura_tile, largura_tile):
            print(
                f"[Aviso] {os.path.basename(caminho)} tem tamanho {img.shape[:2]}, "
                f"esperado {(altura_tile, largura_tile)} -- pulando."
            )
            continue

        # Descarta a faixa que o tile vizinho também fotografou.
        img = img[:recorte_y, :recorte_x]

        if (recorte_y, recorte_x) != (altura, largura):
            # INTER_AREA é a interpolação certa pra reduzir: faz a média dos
            # pixels descartados, em vez de simplesmente pular alguns.
            img = cv2.resize(img, (largura, altura), interpolation=cv2.INTER_AREA)

        # A moldura é desenhada DEPOIS do resize, de propósito: a espessura
        # é medida em pixels do mosaico final, não da imagem original.
        if (i, j) in posicoes_com_flake:
            desenhar_destaque(img, cor_destaque)

        # i (eixo X do estágio) vira coluna, j (eixo Y) vira linha -- mesma
        # convenção do hstack/vstack que a varredura usava antes.
        mosaico[j * altura:(j + 1) * altura, i * largura:(i + 1) * largura] = img

        if contador % 50 == 0 or contador == total:
            print(f"  {contador}/{total} imagens encaixadas...")

    # flush força o sistema operacional a gravar em disco o que ainda estiver
    # em buffer -- sem isso o arquivo pode ficar incompleto.
    mosaico.flush()

    print(f"\nSucesso! Mosaico salvo em: {caminho_saida}")


if __name__ == "__main__":
    main()
