"""
Levanta a curva da razão de foco contra Z.

Para que serve: converter a razão de um número relativo em escala física. É daqui
que sai o RAZAO_MINIMA -- sem esta curva, qualquer limiar é chute.

Testa três afirmações de uma vez, e as três precisam se confirmar:
  - a razão tem PICO no foco, e o pico é afiado o bastante para servir
    de gatilho perto do ótimo;
  - a nitidez pica no MESMO Z (se discordarem, uma das duas está errada);
  - a textura fica quase PLANA (é o que a docstring dela promete e nunca
    foi verificado).

Como usar:
  1. Objetiva da DETECÇÃO montada.
  2. Estágio num campo COM ESTRUTURA. Campo limpo não serve: a curva sai
     plana e não mede nada.
  3. Foque à mão o melhor que conseguir.
  4. Rode. Só o Z se move.

Salva o stack inteiro em disco de propósito: com as imagens guardadas,
ajustar os SIGMA_* e refazer a análise vira um laço offline de segundos,
sem voltar ao microscópio.
"""

import csv
import datetime
import os

import numpy as np
from pycromanager import Core
from tifffile import imwrite

from caminhos import DIRETORIO_SAIDA_BASE
from capturar_imagem import capturar_imagem
from comandos import medir_foco_relativo, medir_nitidez, medir_textura
from parametros import (
    EXPOSICAO_DETECCAO,
    WHITE_BALANCE_BLUE,
    WHITE_BALANCE_RED,
)

# Meia-faixa varrida, em µm. 10 é generoso de propósito: é melhor sobrar
# curva dos dois lados do que descobrir que o pico ficou fora do alcance.
FAIXA_UM = 10.0

# Mesmo passo do autofocar, para os dois falarem a mesma língua.
PASSO_UM = 0.5

# Abaixo disto o campo não tem estrutura suficiente e a curva mediria ruído.
TEXTURA_MINIMA_STACK = 3.0


def main():
    core = Core()

    camera = core.get_camera_device()
    z_stage = core.get_focus_device()

    if core.is_sequence_running():
        core.stop_sequence_acquisition()

    # As mesmas condições da varredura de detecção. Sem isso a curva seria
    # levantada com um brilho/ruído diferente do que o gatilho vai ver na
    # prática, e o limiar não transferiria.
    core.set_exposure(EXPOSICAO_DETECCAO)
    core.set_property(camera, "WhiteBalanceRed", WHITE_BALANCE_RED)
    core.set_property(camera, "WhiteBalanceBlue", WHITE_BALANCE_BLUE)

    z0 = core.get_position(z_stage)
    print(f"Z inicial (o que você focou à mão): {z0:.3f}")

    # Confere o campo ANTES de gastar 41 capturas nele.
    textura_inicial = medir_textura(capturar_imagem(core, camera))

    if textura_inicial < TEXTURA_MINIMA_STACK:
        raise SystemExit(
            f"Campo sem estrutura suficiente (textura {textura_inicial:.2f} < "
            f"{TEXTURA_MINIMA_STACK}). Mova para um campo com flake, risco ou "
            "borda e rode de novo."
        )

    print(f"Textura do campo: {textura_inicial:.2f} -- ok\n")

    timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    diretorio = os.path.join(DIRETORIO_SAIDA_BASE, "ZStack_" + timestamp)
    os.makedirs(diretorio, exist_ok=True)

    # Varre num sentido só, do menor Z para o maior. Inverter o sentido no
    # meio traria a folga mecânica do estágio (backlash) para dentro da
    # medida, e a curva sairia assimétrica por um motivo que não tem nada
    # a ver com óptica.
    deltas = np.arange(-FAIXA_UM, FAIXA_UM + PASSO_UM, PASSO_UM)

    razoes = []
    nitidezes = []
    texturas = []

    try:
        for indice, delta in enumerate(deltas):
            core.set_position(z_stage, z0 + float(delta))
            core.wait_for_device(z_stage)

            imagem = capturar_imagem(core, camera)

            imwrite(
                os.path.join(diretorio, f"stack_{indice:03d}_dz{delta:+.2f}.tif"),
                imagem,
            )

            razoes.append(medir_foco_relativo(imagem))
            nitidezes.append(medir_nitidez(imagem))
            texturas.append(medir_textura(imagem))

            print(f"  dz={delta:+6.2f} µm  razao={razoes[-1]:.4f}")

    finally:
        # Devolve o estágio ao ponto de partida aconteça o que acontecer.
        print("\nRetornando Z à posição inicial...")
        core.set_position(z_stage, z0)
        core.wait_for_device(z_stage)

    razoes = np.array(razoes)
    nitidezes = np.array(nitidezes)
    texturas = np.array(texturas)

    caminho_csv = os.path.join(diretorio, "curva_foco.csv")

    with open(caminho_csv, mode='w', newline='', encoding='utf-8-sig') as arquivo:
        escritor = csv.writer(arquivo, delimiter=';')
        escritor.writerow(["dz_um", "Razao", "Nitidez", "Textura"])
        for delta, r, n, t in zip(deltas, razoes, nitidezes, texturas):
            escritor.writerow([f"{delta:+.2f}", f"{r:.4f}", f"{n:.2f}", f"{t:.3f}"])

    # --- CURVA NORMALIZADA ---
    # Cada métrica dividida pelo próprio máximo. É assim que dá para
    # comparar as três no mesmo eixo: o que interessa é a FORMA de cada
    # uma, não a escala.
    print("\n--- CURVA (cada coluna dividida pelo próprio máximo) ---")
    print("  dz(µm)    razao   razao/max   nitidez  nit/max   textura  tex/max")

    for delta, r, n, t in zip(deltas, razoes, nitidezes, texturas):
        print(
            f"  {delta:+6.2f}  {r:7.4f}   {r / razoes.max():7.3f}   "
            f"{n:8.1f}  {n / nitidezes.max():6.3f}   "
            f"{t:7.2f}  {t / texturas.max():6.3f}"
        )

    pico_razao = int(np.argmax(razoes))
    pico_nitidez = int(np.argmax(nitidezes))

    print("\n--- RESUMO ---")
    print(f"pico da RAZAO   em dz = {deltas[pico_razao]:+.2f} µm  (razao {razoes[pico_razao]:.4f})")
    print(f"pico da NITIDEZ em dz = {deltas[pico_nitidez]:+.2f} µm")
    print(f"variação da TEXTURA na faixa: {texturas.min() / texturas.max():.2f} a 1.00")

    if pico_razao in (0, len(deltas) - 1):
        print(
            "\n[AVISO] O pico caiu na PONTA da faixa: o foco real está fora do\n"
            "        que foi varrido. Refocalize à mão e rode de novo -- esta\n"
            "        curva não serve para calibrar."
        )
        return

    # Largura da curva: quantos µm até a razão cair a uma dada fração do
    # pico. É isto que dá a tradução razão <-> µm, e a meia-largura é a
    # profundidade de campo operacional da objetiva.
    print("\n--- QUEDA DA RAZÃO A PARTIR DO PICO ---")

    for fracao in (0.90, 0.70, 0.50):
        alvo = razoes[pico_razao] * fracao

        esquerda = next(
            (deltas[k] for k in range(pico_razao, -1, -1) if razoes[k] <= alvo),
            None,
        )
        direita = next(
            (deltas[k] for k in range(pico_razao, len(deltas)) if razoes[k] <= alvo),
            None,
        )

        e = f"{esquerda:+.2f}" if esquerda is not None else "fora da faixa"
        d = f"{direita:+.2f}" if direita is not None else "fora da faixa"

        print(f"  {int(fracao * 100)}% do pico (razao {alvo:.4f}):  dz = {e} / {d} µm")

    print(f"\nImagens e CSV em: {diretorio}")


if __name__ == "__main__":
    main()