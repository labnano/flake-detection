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
  2. Estágio num campo COM ESTRUTURA. Campo limpo não serve: a curva saiplana e não mede nada.
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
from comandos import escolher_objetiva, medir_foco_relativo, medir_nitidez, medir_textura

# Este script precisa da exposição e do balanço de branco da objetiva -- não
# do fov. A curva tem que ser levantada com o MESMO brilho que a varredura
# vai usar, senão a nitidez e a textura saem em outra escala e nada do que
# se calibrar aqui transfere.
CAMPOS_NECESSARIOS = ("exposicao", "wb_red", "wb_blue")

# Meia-faixa e passo do stack, SUGERIDOS por objetiva.
#
# Eles têm de acompanhar a profundidade de campo, que cai com o quadrado da
# abertura numérica -- e a NA cresce com a magnificação. Uma faixa de ±10 µm
# é generosa no 50x e ESTREITA DEMAIS no 5x: lá a DOF sozinha passa de 20 µm,
# então a curva inteira sairia dentro do pico, chapada, sem largura nenhuma
# para medir.
#
# São chutes de partida, calculados de DOF ~ lambda/NA² com as NA típicas de
# cada magnificação, varrendo ~3x a DOF para cada lado em passos de ~DOF/5.
# O script pergunta e você pode mudar: se o pico sair na ponta, alargue; se
# sair chapado, alargue também (e engrosse o passo, senão são fotos demais).
SUGESTAO_STACK = {
    "5x": (60.0, 4.0),
    "10x": (25.0, 1.5),
    "20x": (10.0, 0.5),
    "50x": (10.0, 0.5),
    "100x": (5.0, 0.25),
}

# Usado só se a objetiva ainda não tiver textura_minima calibrada.
TEXTURA_MINIMA_PADRAO = 3.0


def perguntar_float(rotulo, padrao):
    """Pergunta um número, aceitando ENTER para ficar com o sugerido."""

    while True:
        entrada = input(f"{rotulo} [ENTER = {padrao}]: ").strip().replace(",", ".")

        if entrada == "":
            return padrao

        try:
            valor = float(entrada)
        except ValueError:
            print("[Erro] Digite um número, ou ENTER para aceitar o sugerido.")
            continue

        if valor <= 0:
            print("[Erro] Tem que ser maior que zero.")
            continue

        return valor


def main():
    core = Core()

    camera = core.get_camera_device()
    z_stage = core.get_focus_device()

    if core.is_sequence_running():
        core.stop_sequence_acquisition()

    nome_objetiva, config = escolher_objetiva(exigidos=CAMPOS_NECESSARIOS)

    # As mesmas condições da varredura. Sem isso a curva seria levantada com
    # um brilho/ruído diferente do que o gatilho vai ver na prática, e nada
    # do que se calibrar aqui transferiria.
    core.set_exposure(config["exposicao"])
    core.set_property(camera, "WhiteBalanceRed", config["wb_red"])
    core.set_property(camera, "WhiteBalanceBlue", config["wb_blue"])

    faixa_sugerida, passo_sugerido = SUGESTAO_STACK.get(nome_objetiva, (10.0, 0.5))

    print(f"\n--- FAIXA DO STACK (objetiva {nome_objetiva}) ---")
    print(
        "Sugestão calculada da profundidade de campo típica desta magnificação.\n"
        "Se o pico sair na ponta da faixa OU sair chapado, alargue e rode de novo."
    )

    faixa_um = perguntar_float("Meia-faixa em µm (varre para os dois lados)", faixa_sugerida)
    passo_um = perguntar_float("Passo em µm", passo_sugerido)

    qtd_fotos = int(2 * faixa_um / passo_um) + 1
    print(f"-> {qtd_fotos} fotos, de {-faixa_um:+.2f} a {faixa_um:+.2f} µm.\n")

    # O portão de textura é o da própria objetiva, se já estiver calibrado
    # (calibrar_textura.py). O padrão só existe para não travar quem ainda
    # não chegou nesse passo -- mas ele foi medido com OUTRO brilho, então
    # vale pouco.
    textura_minima = config.get("textura_minima")

    if textura_minima is None:
        textura_minima = TEXTURA_MINIMA_PADRAO
        print(
            f"[Aviso] A objetiva {nome_objetiva} ainda não tem textura_minima; usando "
            f"{TEXTURA_MINIMA_PADRAO} como\n        portão provisório. Ele foi medido "
            "com outra exposição e serve só para\n        barrar campo obviamente vazio."
        )

    z0 = core.get_position(z_stage)
    print(f"Z inicial (o que você focou à mão): {z0:.3f}")

    # Confere o campo ANTES de gastar dezenas de capturas nele.
    textura_inicial = medir_textura(capturar_imagem(core, camera))

    if textura_inicial < textura_minima:
        raise SystemExit(
            f"Campo sem estrutura suficiente (textura {textura_inicial:.2f} < "
            f"{textura_minima}). Mova para um campo com flake, risco ou "
            "borda e rode de novo."
        )

    print(f"Textura do campo: {textura_inicial:.2f} -- ok\n")

    timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    diretorio = os.path.join(
        DIRETORIO_SAIDA_BASE, f"ZStack_{nome_objetiva}_{timestamp}"
    )
    os.makedirs(diretorio, exist_ok=True)

    # Varre num sentido só, do menor Z para o maior. Inverter o sentido no
    # meio traria a folga mecânica do estágio (backlash) para dentro da
    # medida, e a curva sairia assimétrica por um motivo que não tem nada
    # a ver com óptica.
    deltas = np.arange(-faixa_um, faixa_um + passo_um, passo_um)

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
            f"  {delta:+6.2f}  {r:7.4f}   {r / np.nanmax(razoes):7.3f}   "
            f"{n:8.1f}  {n / nitidezes.max():6.3f}   "
            f"{t:7.2f}  {t / texturas.max():6.3f}"
        )

    # nanargmax, e não argmax: a medir_foco_relativo devolve NaN de
    # propósito quando o campo não tem denominador -- o que acontece
    # justamente nas pontas de um stack largo, onde o desfoque apagou até a
    # estrutura grossa. O np.argmax devolveria o índice do PRIMEIRO NaN, e
    # o script concluiria "pico na ponta da faixa" numa curva perfeitamente
    # boa. Foi assim que este bug apareceu: só com stack largo, que é o
    # caso das objetivas de baixa magnificação.
    quantidade_nan = int(np.isnan(razoes).sum())

    if quantidade_nan == len(razoes):
        print(
            "\n[AVISO] A razão não pôde ser medida em NENHUMA posição do stack "
            "(campo\n        uniforme demais). Escolha um campo com mais estrutura."
        )
        return

    if quantidade_nan:
        print(
            f"\n[Nota] Em {quantidade_nan} das {len(razoes)} posições a razão saiu NaN: "
            "ali o\n       desfoque apagou até a estrutura grossa, e não há denominador. "
            "É\n       informação, não erro -- só significa que a faixa cobre bem o pico."
        )

    pico_razao = int(np.nanargmax(razoes))
    pico_nitidez = int(np.argmax(nitidezes))

    print("\n--- RESUMO ---")
    print(f"pico da RAZAO   em dz = {deltas[pico_razao]:+.2f} µm  (razao {razoes[pico_razao]:.4f})")
    print(f"pico da NITIDEZ em dz = {deltas[pico_nitidez]:+.2f} µm")
    print(f"variação da TEXTURA na faixa: {texturas.min() / texturas.max():.2f} a 1.00")

    if pico_razao in (0, len(deltas) - 1):
        print(
            "\n[AVISO] O pico caiu na PONTA da faixa: o foco real está fora do\n"
            "        que foi varrido. Esta curva não serve para calibrar.\n"
            "        Duas causas, e as duas se resolvem rodando de novo:\n"
            "        (a) o foco manual estava longe -- refocalize;\n"
            f"        (b) a meia-faixa de {faixa_um} µm é estreita para esta objetiva --\n"
            "            dobre-a (e dobre o passo junto, senão são fotos demais)."
        )
        return

    # Largura da curva: quantos µm até a razão cair a uma dada fração do
    # pico. É isto que dá a tradução razão <-> µm, e a meia-largura é a
    # profundidade de campo operacional da objetiva.
    print("\n--- QUEDA DA RAZÃO A PARTIR DO PICO ---")

    larguras = {}

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

        if esquerda is not None and direita is not None:
            larguras[fracao] = float(direita - esquerda)
            print(
                f"  {int(fracao * 100)}% do pico (razao {alvo:.4f}):  dz = {e} / {d} µm"
                f"   -> largura {larguras[fracao]:.2f} µm"
            )
        else:
            print(f"  {int(fracao * 100)}% do pico (razao {alvo:.4f}):  dz = {e} / {d} µm")

    # --- O QUE COLAR NO parametros.py ---
    #
    # A largura a 90% do pico é a profundidade de campo OPERACIONAL: a faixa
    # de Z dentro da qual a imagem é praticamente tão nítida quanto no foco.
    # Ela é o que dimensiona a busca do autofoco.
    print("\n--- PARÂMETROS DE AUTOFOCO SUGERIDOS ---")

    if 0.90 not in larguras:
        print(
            "  A curva não caiu a 90% do pico dentro da faixa varrida, então a\n"
            "  profundidade de campo é MAIOR que o que foi varrido e não dá para\n"
            f"  medi-la. Rode de novo com meia-faixa bem maior que {faixa_um} µm."
        )
    else:
        dof = larguras[0.90]

        # passo ~ DOF/4: o mínimo para o pico não cair entre duas amostras
        # (é o Nyquist da busca). faixa ~ 2,5 x DOF: larga o bastante para
        # abraçar o erro do modelo de foco, estreita o bastante para o
        # checkpoint não custar caro -- ele gasta 2*faixa/passo fotos.
        passo_sugerido_af = dof / 4.0
        faixa_sugerida_af = dof * 2.5
        fotos_checkpoint = int(2 * faixa_sugerida_af / passo_sugerido_af) + 1

        print(f"  Profundidade de campo medida (largura a 90%): {dof:.2f} µm")
        print(f"  Cada checkpoint de autofoco custará {fotos_checkpoint} fotos.")
        print("\n  Cole no parametros.py:")
        print(f'    OBJETIVAS["{nome_objetiva}"]["autofoco_faixa_um"] = {faixa_sugerida_af:.2f}')
        print(f'    OBJETIVAS["{nome_objetiva}"]["autofoco_passo_um"] = {passo_sugerido_af:.2f}')

    print(f"\nImagens e CSV em: {diretorio}")


if __name__ == "__main__":
    main()