"""Varredura de uma ou mais amostras, para montar mosaico e para ajustar
os parâmetros de detecção offline.

COMO O SCRIPT ESTÁ ORGANIZADO
-----------------------------
Ele é dividido em quatro fases, e a divisão não é cosmética: tudo que exige
você presente acontece ANTES de qualquer foto ser tirada, para que a parte
demorada rode sozinha.

    Fase 0  perguntas gerais: objetiva, quantas amostras, modelo de foco,
            quantos pontos de calibração. Uma vez só, valendo para todas.
    Fase 1  os 4 cantos de cada amostra. É a última chance de realinhar um
            chip torto no suporte -- depois disso, mexer no suporte
            invalidaria os pontos de foco.
    Fase 2  os pontos de foco de cada amostra, guiados: o script leva o
            estágio até a posição e você só ajusta o foco. O modelo é
            ajustado JÁ, aqui, e não na hora de varrer -- assim uma
            geometria ruim aparece enquanto você ainda está na frente do
            microscópio.
    Fase 3  resumo de tudo e um único ENTER autorizando.
    Fase 4  varredura, desassistida.

O QUE SAI EM DISCO
------------------
    Varredura_<timestamp>/
        lote.json               o que valeu para a rodada inteira
        tempos.json             onde o tempo foi gasto, por etapa
        Amostra_1/
            img_pos_X0_Y0.tif   ...um por posição da grade
            varredura.json      geometria, objetiva, cantos, modelo de foco
            autofoco.csv        o que cada checkpoint de foco mediu
        Amostra_2/
            ...

Cada pasta de amostra é autossuficiente: o montar_mosaico.py roda direto
nela, e o autofoco.csv é o que calibra a textura_minima daquela objetiva.

    python montar_mosaico.py "<...>/Varredura_<timestamp>/Amostra_1"
"""

import csv
import datetime
import json
import os
import time

import numpy as np
from tifffile import imwrite
from pycromanager import Core

from caminhos import DIRETORIO_SAIDA_BASE
from parametros import SOBREPOSICAO, CAMPOS_AUTOFOCO
from capturar_imagem import capturar_imagem
from comandos import (
    FATOR_PASSO_GROSSO,
    QTD_PONTOS_CONFERENCIA,
    Cronometro,
    calibrar_plano_foco,
    calibrar_superficie_foco,
    calcular_z,
    campos_faltando,
    checkpoint_autofoco,
    coletar_pontos_calibracao,
    definir_area_amostra,
    escolher_objetiva,
    gerar_malha,
    ordem_serpentina,
    perguntar_qtd_pontos,
    reancorar_amostra,
    salvar_metadados_varredura,
)

# --- CONFIGURAÇÕES DE AUTOFOCO ---
# A cada INTERVALO_AUTOFOCO imagens o script procura o melhor Z ao redor do
# valor previsto pelo modelo. A FAIXA e o PASSO da busca não estão aqui:
# eles dependem da profundidade de campo e por isso moram no parametros.py,
# por objetiva.
#
# Não há mais um checkpoint no início de cada coluna. Ele existia porque a
# varredura em raster voltava do fim de uma coluna para o começo da
# seguinte, e era ali que o foco mais se perdia; com a varredura em
# serpentina (ver ordem_serpentina) esse salto deixou de existir.
INTERVALO_AUTOFOCO = 20

# Quantas fotos um checkpoint custa, TIPICAMENTE, com a busca expansiva --
# usado só para estimar a duração no resumo da Fase 3.
#
# O mínimo é 3 (centro e os dois vizinhos, quando o modelo acerta) e o
# máximo é 2*faixa/passo + 1, o custo da varredura completa da faixa, que
# só acontece quando o modelo erra o suficiente para a busca ter de
# atravessar a faixa inteira. 5 é um meio-termo honesto; a estimativa
# imprime também o pior caso, para não vender otimismo.
FOTOS_POR_CHECKPOINT_TIPICO = 5

# Quanto a faixa de busca é alargada no PRIMEIRO checkpoint de cada amostra.
#
# Motivo: mesmo depois da re-ancoragem (ver REANCORAGEM_FAIXA_UM), o foco
# continua andando enquanto a própria amostra é varrida. A primeira busca
# ainda é a que tem mais chance de errar, porque é a única sem nenhum
# checkpoint anterior para se apoiar. Abrir a busca uma vez, no começo,
# custa algumas fotos.
#
# Este número era a ÚNICA defesa contra a deriva acumulada desde a Fase 2, e
# a varredura de 17/08 mostrou que ele não dava conta: com deriva de
# ~35 µm/h e amostras esperando até 7 h, ±30 µm não chegava nem perto. Quem
# cuida disso agora é a re-ancoragem, que mede o deslocamento em vez de
# tentar adivinhá-lo.
FATOR_FAIXA_ABERTURA = 3.0

# Faixa da busca GROSSA da re-ancoragem, em µm -- ver reancorar_amostra.
#
# Ela precisa cobrir o quanto o foco anda entre a Fase 2 e a varredura da
# ÚLTIMA amostra da rodada, que é o pior caso. O valor certo é
# (deriva por hora) x (duração da rodada), com margem: na varredura de
# 17/08 foram ~35 µm/h por 8,9 h, ou seja ~310 µm.
#
# O custo é uma varredura exaustiva, 2*faixa/passo + 1 fotos, uma vez por
# amostra. Na 50x (passo grosso de 2 µm) isso são 301 fotos, ~2 min --
# contra as ~1,7 h que a amostra leva para ser varrida.
#
# Encolher isto é a primeira economia possível DEPOIS de medir a deriva do
# seu equipamento; enquanto ela não for medida, errar para o lado largo
# custa dois minutos e errar para o estreito custa a amostra inteira.
REANCORAGEM_FAIXA_UM = 300.0

# Chute inicial de segundos por imagem, só para a estimativa de duração do
# resumo. Durante a varredura a estimativa passa a usar o tempo medido.
TEMPO_POR_IMAGEM_ESTIMADO = 1.5


# =====================================================================
#  FASE 0 -- perguntas gerais
# =====================================================================

core = Core()

camera = core.get_camera_device()
xy_stage = core.get_xy_stage_device()
z_stage = core.get_focus_device()

# Posição de onde o estágio saiu, para voltar aqui no fim aconteça o que
# acontecer -- inclusive se a varredura quebrar no meio.
x_origem = core.get_x_position(xy_stage)
y_origem = core.get_y_position(xy_stage)

nome_objetiva, config_objetiva = escolher_objetiva()

fov_x, fov_y = config_objetiva["fov"]

# A exposição e o balanço de branco são aplicados AGORA, antes das fases
# manuais, e não só antes de varrer: assim o que você vê no live enquanto
# marca cantos e ajusta foco é a mesma imagem que a varredura vai gravar.
core.set_exposure(config_objetiva["exposicao"])
core.set_property(camera, "WhiteBalanceRed", config_objetiva["wb_red"])
core.set_property(camera, "WhiteBalanceBlue", config_objetiva["wb_blue"])

# O autofoco é opcional: sem os três campos calibrados, dá para varrer
# confiando só no modelo ajustado na Fase 2. Pior, mas honesto -- e é o
# único jeito de fazer a PRIMEIRA varredura de uma objetiva nova, já que a
# textura_minima dela só pode ser medida a partir de uma varredura.
faltam_autofoco = campos_faltando(nome_objetiva, CAMPOS_AUTOFOCO)

if faltam_autofoco:
    print(
        f"[Aviso] A objetiva {nome_objetiva} não tem autofoco calibrado: "
        f"falta {', '.join(faltam_autofoco)}.\n"
        "        Sem isso a varredura roda confiando apenas no modelo de foco\n"
        "        ajustado no início -- qualquer deriva ao longo da amostra fica\n"
        "        sem correção. O autofoco.csv sai mesmo assim, com a textura de\n"
        "        cada posição, que é o que calibra a textura_minima."
    )

    resposta = input("Varrer assim mesmo, sem autofoco? [s/N]: ").strip().lower()

    if resposta != "s":
        raise SystemExit(
            "Cancelado. Calibre a objetiva (calibrar_razao_foco.py para a faixa "
            "e o passo) e rode de novo."
        )

    config_autofoco = None

else:
    config_autofoco = {
        "faixa_um": config_objetiva["autofoco_faixa_um"],
        "passo_um": config_objetiva["autofoco_passo_um"],
        "textura_minima": config_objetiva["textura_minima"],
    }

while True:
    try:
        qtd_amostras = int(input("\nQuantas amostras serão varridas nesta rodada? "))
    except ValueError:
        print("[Erro] Digite um número inteiro.")
        continue

    if qtd_amostras < 1:
        print("[Erro] Pelo menos 1 amostra.")
        continue

    break

print("\n--- MÉTODO DE CALIBRAÇÃO DE FOCO ---")
print("1. Plano (mais rápido, funciona bem se a amostra for praticamente plana)")
print("2. Superfície curva/quadrática (mais precisa com magnificações maiores, exige mais pontos)")
print("   Vale para TODAS as amostras desta rodada.")

while True:
    metodo = input("Escolha o método (1 ou 2): ").strip()

    if metodo == "1":
        tipo_modelo = "plano"
        minimo_pontos = 3
        break

    if metodo == "2":
        # A quadrática tem 6 coeficientes, então 6 pontos é o mínimo
        # matemático -- mas com exatamente 6 o ajuste passa por todos eles
        # e reproduz o erro de foco de cada um sem nenhuma média para
        # amortecer. Medido: com 8 pontos o erro máximo ainda fica em ~2 µm;
        # com 10 cai para ~0,9 µm; com 12, ~0,65 µm. Daí o mínimo de 10.
        tipo_modelo = "quadratica"
        minimo_pontos = 10
        break

    print("[Erro] Digite 1 ou 2.\n")

print(f"\nEsse método precisa de pelo menos {minimo_pontos} pontos por amostra.")
qtd_pontos = perguntar_qtd_pontos(minimo_pontos)


# =====================================================================
#  FASE 1 -- os 4 cantos de cada amostra
# =====================================================================

print("\n" + "=" * 70)
print(f"FASE 1 de 3 -- CANTOS DAS {qtd_amostras} AMOSTRA(S)")
print("=" * 70)
print(
    "\nA área varrida é o maior retângulo RETO que cabe dentro dos 4 cantos\n"
    "(a área azul do seu desenho). Nada é fotografado fora da amostra; em\n"
    "troca, quanto mais torto o chip, mais beirada fica de fora."
)

amostras = []

for numero in range(1, qtd_amostras + 1):
    rotulo = f"[Amostra {numero}/{qtd_amostras}] "

    print("\n" + "-" * 70)
    print(f"AMOSTRA {numero} de {qtd_amostras}")
    print("-" * 70)

    cantos, caixa = definir_area_amostra(core, xy_stage, rotulo)

    malha = gerar_malha(
        caixa["x_min"],
        caixa["y_min"],
        caixa["x_max"],
        caixa["y_max"],
        fov_x,
        fov_y,
        SOBREPOSICAO,
        # A explicação da sobreposição é a mesma para todas as amostras --
        # imprimir uma vez informa, cinco vezes só empurra o resto pra cima.
        verboso=(numero == 1),
    )

    print(
        f"  Grade: {malha['qtd_passos_x']} x {malha['qtd_passos_y']} = "
        f"{malha['qtd_passos_x'] * malha['qtd_passos_y']} imagens"
    )

    amostras.append(
        {
            "numero": numero,
            "cantos": cantos,
            "caixa": caixa,
            "malha": malha,
        }
    )


# =====================================================================
#  FASE 2 -- pontos de foco de cada amostra
# =====================================================================

print("\n" + "=" * 70)
print("FASE 2 de 3 -- PONTOS DE FOCO")
print("=" * 70)
print(
    "\nCada amostra tem seu PRÓPRIO modelo de foco, independente das outras:\n"
    "chips diferentes assentam em alturas diferentes no suporte, e misturar\n"
    "os pontos de duas amostras num plano só seria um erro grosseiro."
)

for amostra in amostras:
    numero = amostra["numero"]

    print("\n" + "-" * 70)
    print(f"AMOSTRA {numero} de {qtd_amostras} -- {qtd_pontos} pontos de foco")
    print("-" * 70)

    # O ajuste acontece aqui, junto da coleta, e não na Fase 4. Se a
    # geometria dos pontos não servir (todos quase alinhados, área
    # alongada demais), você descobre agora e refaz -- e não daqui a duas
    # horas, com a varredura já rodando e você longe do microscópio.
    while True:
        pontos = coletar_pontos_calibracao(
            core,
            xy_stage,
            z_stage,
            minimo_pontos,
            malha=amostra["malha"],
            tipo=tipo_modelo,
            qtd_pontos=qtd_pontos,
        )

        try:
            if tipo_modelo == "quadratica":
                modelo_foco = calibrar_superficie_foco(pontos)
            else:
                modelo_foco = calibrar_plano_foco(pontos)

            break

        except (ValueError, np.linalg.LinAlgError) as erro:
            print(f"\n[Erro] {erro}")
            print(
                f"\nO ajuste da amostra {numero} não fechou. Vamos coletar os "
                "pontos dela de novo -- espalhe-os melhor pela área."
            )

    amostra["pontos_foco"] = pontos
    amostra["modelo_foco"] = modelo_foco

    valores_z = [
        calcular_z(n, m, modelo_foco)
        for n in amostra["malha"]["malha_x"]
        for m in amostra["malha"]["malha_y"]
    ]

    amostra["z_minimo"] = min(valores_z)
    amostra["z_maximo"] = max(valores_z)

    print(f"\n[Sucesso] Modelo de foco da amostra {numero} calculado.")


# =====================================================================
#  FASE 3 -- resumo e confirmação
# =====================================================================

timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
diretorio_lote = os.path.join(DIRETORIO_SAIDA_BASE, "Varredura_" + timestamp)
os.makedirs(diretorio_lote, exist_ok=True)

for amostra in amostras:
    amostra["diretorio"] = os.path.join(diretorio_lote, f"Amostra_{amostra['numero']}")

total_imagens = sum(
    amostra["malha"]["qtd_passos_x"] * amostra["malha"]["qtd_passos_y"]
    for amostra in amostras
)

# Custo do autofoco: um checkpoint a cada INTERVALO_AUTOFOCO imagens, cada
# um custando entre 3 fotos (modelo certo) e 2*faixa/passo + 1 (a faixa
# inteira, quando o modelo erra muito). A estimativa mostra os dois.
fotos_por_checkpoint = 0
fotos_por_checkpoint_maximo = 0

# Custo da re-ancoragem, por amostra: a varredura exaustiva da busca grossa
# (que é o grosso da conta) mais a busca fina e as conferências, que custam
# poucas fotos cada quando o deslocamento medido está certo.
fotos_reancoragem = 0

if config_autofoco:
    fotos_por_checkpoint = FOTOS_POR_CHECKPOINT_TIPICO
    fotos_por_checkpoint_maximo = int(
        2 * config_autofoco["faixa_um"] / config_autofoco["passo_um"] + 1
    )

    passo_grosso = max(
        config_autofoco["passo_um"],
        FATOR_PASSO_GROSSO * config_autofoco["faixa_um"],
    )
    fotos_reancoragem = (
        int(2 * REANCORAGEM_FAIXA_UM / passo_grosso) + 1
        + (1 + QTD_PONTOS_CONFERENCIA) * FOTOS_POR_CHECKPOINT_TIPICO
    )

checkpoints = sum(
    amostra["malha"]["qtd_passos_x"] * amostra["malha"]["qtd_passos_y"] // INTERVALO_AUTOFOCO
    for amostra in amostras
)

fotos_extras = checkpoints * fotos_por_checkpoint + qtd_amostras * fotos_reancoragem

segundos = (total_imagens + fotos_extras) * TEMPO_POR_IMAGEM_ESTIMADO
segundos_maximo = (
    total_imagens
    + checkpoints * fotos_por_checkpoint_maximo
    + qtd_amostras * fotos_reancoragem
) * TEMPO_POR_IMAGEM_ESTIMADO

print("\n" + "=" * 70)
print("FASE 3 de 3 -- RESUMO")
print("=" * 70)

print(f"\nObjetiva: {nome_objetiva}  (campo de {fov_x:.1f} x {fov_y:.1f} µm)")
print(f"Exposição: {config_objetiva['exposicao']} ms  |  WB R/B: "
      f"{config_objetiva['wb_red']}/{config_objetiva['wb_blue']}")
print(f"Sobreposição: {SOBREPOSICAO:.0%}  |  Modelo de foco: {tipo_modelo}")

if config_autofoco:
    print(
        f"Autofoco: +-{config_autofoco['faixa_um']} µm em passos de "
        f"{config_autofoco['passo_um']} µm, busca do centro para fora "
        f"(3 fotos se o modelo acertar, {fotos_por_checkpoint_maximo} no pior caso), "
        f"textura mínima {config_autofoco['textura_minima']}"
    )
    print(
        f"Re-ancoragem: +-{REANCORAGEM_FAIXA_UM:.0f} µm (~{fotos_reancoragem} fotos, "
        f"~{fotos_reancoragem * TEMPO_POR_IMAGEM_ESTIMADO / 60:.0f} min) no começo de "
        "cada amostra,\n"
        "              para medir o quanto o foco andou desde a Fase 2."
    )
else:
    print("Autofoco: DESLIGADO (objetiva sem calibração de foco)")
    print(
        "Re-ancoragem: DESLIGADA junto -- ela usa a mesma busca de foco. Os pontos da\n"
        "              Fase 2 valerão como foram coletados, por mais tempo que passe\n"
        "              até cada amostra ser varrida."
    )

print(f"\n{'Amostra':<9} {'Grade':>11} {'Imagens':>9} {'Área (µm)':>17} {'Rot.':>7} {'Perda':>7} {'Z (µm)':>19}")

for amostra in amostras:
    malha = amostra["malha"]
    caixa = amostra["caixa"]
    imagens = malha["qtd_passos_x"] * malha["qtd_passos_y"]

    print(
        f"{amostra['numero']:<9} "
        f"{malha['qtd_passos_x']:>4} x {malha['qtd_passos_y']:<4} "
        f"{imagens:>9} "
        f"{caixa['largura']:>7.0f} x {caixa['altura']:<7.0f} "
        f"{caixa['rotacao_graus']:>+6.1f}° "
        f"{caixa['perda_area']:>6.0%} "
        f"{amostra['z_minimo']:>8.2f} a {amostra['z_maximo']:<8.2f}"
    )

def formatar_duracao(segundos):
    return f"{segundos / 3600:.1f} h" if segundos >= 3600 else f"{segundos / 60:.0f} min"


print(f"\nTotal: {total_imagens} imagens em {qtd_amostras} amostra(s)")
print(
    f"Duração estimada: ~{formatar_duracao(segundos)}, no pior caso "
    f"{formatar_duracao(segundos_maximo)}  (grosseira: supõe "
    f"{TEMPO_POR_IMAGEM_ESTIMADO} s por foto, e {checkpoints} checkpoints de "
    f"{fotos_por_checkpoint} a {fotos_por_checkpoint_maximo} fotos)"
)
print(
    "    A varredura mede o tempo de verdade enquanto roda e imprime, no fim de\n"
    "    cada amostra, onde ele foi gasto (estágio XY, estágio Z, câmera, disco)."
)
print(f"Destino: {diretorio_lote}")

print(
    "\n[Confira] A objetiva informada foi a "
    f"{nome_objetiva}. O Micro-Manager não tem como conferir isso -- se estiver\n"
    "          errada, o passo do estágio sai errado e a varredura inteira fica\n"
    "          inutilizável, sem nenhum sintoma visível durante a corrida."
)

if qtd_amostras > 1:
    print(
        "\n[Atenção] Entre uma amostra e outra o estágio faz um deslocamento longo\n"
        "          em XY sem recuar o Z. Confirme que as amostras estão em alturas\n"
        "          parecidas no suporte antes de deixar rodando sozinho."
    )

input("\nSe estiver tudo certo, aperte ENTER para iniciar a varredura...")

# Live mode só atrapalha daqui pra frente: as fases manuais acabaram e o
# snap_image precisa da câmera livre.
if core.is_sequence_running():
    core.stop_sequence_acquisition()

# O lote.json descreve a RODADA; o varredura.json dentro de cada pasta
# descreve aquela amostra. Ele é escrito à mão, e não pela
# salvar_metadados_varredura, justamente porque não existe "a grade do
# lote": cada amostra tem a sua. Reaproveitar a função aqui gravaria a
# geometria da amostra 1 como se valesse para a rodada inteira.
with open(os.path.join(diretorio_lote, "lote.json"), "w", encoding="utf-8") as arquivo:
    json.dump(
        {
            "timestamp": timestamp,
            "objetiva": nome_objetiva,
            "fov_x": fov_x,
            "fov_y": fov_y,
            "exposicao_ms": config_objetiva["exposicao"],
            "wb_red": config_objetiva["wb_red"],
            "wb_blue": config_objetiva["wb_blue"],
            "sobreposicao": SOBREPOSICAO,
            "tipo_modelo_foco": tipo_modelo,
            "qtd_pontos_calibracao": qtd_pontos,
            "autofoco": config_autofoco,
            "intervalo_autofoco": INTERVALO_AUTOFOCO,
            "fator_faixa_abertura": FATOR_FAIXA_ABERTURA,
            "reancoragem_faixa_um": REANCORAGEM_FAIXA_UM if config_autofoco else None,
            "qtd_amostras": qtd_amostras,
            "total_imagens": total_imagens,
            "amostras": [
                {
                    "numero": amostra["numero"],
                    "pasta": os.path.basename(amostra["diretorio"]),
                    "grid_x": amostra["malha"]["qtd_passos_x"],
                    "grid_y": amostra["malha"]["qtd_passos_y"],
                    "x0": amostra["malha"]["malha_x"][0],
                    "y0": amostra["malha"]["malha_y"][0],
                    "largura_um": amostra["caixa"]["largura"],
                    "altura_um": amostra["caixa"]["altura"],
                    "rotacao_graus": amostra["caixa"]["rotacao_graus"],
                    "perda_area": amostra["caixa"]["perda_area"],
                    "z_minimo_previsto": amostra["z_minimo"],
                    "z_maximo_previsto": amostra["z_maximo"],
                }
                for amostra in amostras
            ],
        },
        arquivo,
        indent=2,
        ensure_ascii=False,
        default=float,
    )


# =====================================================================
#  FASE 4 -- varredura
# =====================================================================

CABECALHO_AUTOFOCO = [
    "Contador", "i", "j", "X", "Y",
    "Z_previsto", "Z_final", "Textura", "Z_medido", "Correcao",
    "Confiavel", "Acao", "Pontos_modelo", "Fotos", "Queda", "Residuo_RMS",
]


def varrer_amostra(amostra):
    """Varre uma amostra inteira. Tudo que ela precisa saber está no dict.

    Devolve (total_de_imagens, cronometro)."""

    diretorio = amostra["diretorio"]
    os.makedirs(diretorio, exist_ok=True)

    malha = amostra["malha"]
    caixa = amostra["caixa"]

    modelo_foco = amostra["modelo_foco"]
    pontos_foco = list(amostra["pontos_foco"])
    ajuste_z = 0.0

    # RE-ANCORAGEM: entre a Fase 2 e este instante o foco andou, e numa
    # rodada de várias amostras esse intervalo chega a horas. Medir agora,
    # com o estágio já livre, é o que impede a amostra de começar a ser
    # varrida a dezenas de µm do foco -- de onde o autofoco não teria como
    # voltar, porque ele procura em torno do que o modelo prevê.
    #
    # Acontece ANTES do salvar_metadados_varredura de propósito: assim o
    # varredura.json guarda o modelo dos dois momentos, e a diferença entre
    # eles é a medida da deriva daquela espera. É o dado que permite, depois
    # de algumas rodadas, dimensionar a REANCORAGEM_FAIXA_UM com fundamento.
    registro_reancoragem = None

    if config_autofoco:
        pontos_reancorados, modelo_reancorado, registro_reancoragem = reancorar_amostra(
            core,
            xy_stage,
            z_stage,
            camera,
            pontos_foco,
            modelo_foco["tipo"],
            config_autofoco,
            REANCORAGEM_FAIXA_UM,
        )

        if modelo_reancorado is not None:
            pontos_foco = pontos_reancorados
            modelo_foco = modelo_reancorado

    # Quantos pontos vieram da coleta manual da Fase 2. Os que o autofoco
    # medir entram DEPOIS destes na lista, e é essa fronteira que o
    # re-ajuste usa para envelhecer os pontos sem deixar a geometria solta
    # (ver pesos_por_idade no comandos.py).
    #
    # A re-ancoragem não mexe nesta contagem: ela ATUALIZA a altura das
    # âncoras, não acrescenta pontos. Continuam sendo os mesmos (x, y), com
    # o mesmo espalhamento que a Fase 2 validou.
    qtd_ancoras = len(pontos_foco)

    # Os limites de Z previstos foram calculados na Fase 2 com o modelo
    # antigo. Se a re-ancoragem mudou o modelo, eles mudaram junto -- e são
    # eles que o varredura.json publica como faixa de trabalho da amostra.
    valores_z = [
        calcular_z(n, m, modelo_foco)
        for n in malha["malha_x"]
        for m in malha["malha_y"]
    ]

    z_minimo = min(valores_z)
    z_maximo = max(valores_z)

    # Memória curta do autofoco, viva pela amostra inteira: é o que permite
    # um checkpoint saber o que o anterior mediu. Uma amostra nova começa
    # com a memória limpa -- chips diferentes não compartilham história.
    estado_foco = {}

    salvar_metadados_varredura(
        diretorio,
        malha,
        fov_x,
        fov_y,
        extras={
            "amostra": amostra["numero"],
            "objetiva": nome_objetiva,
            "exposicao_ms": config_objetiva["exposicao"],
            "wb_red": config_objetiva["wb_red"],
            "wb_blue": config_objetiva["wb_blue"],
            "cantos_marcados": [[x, y] for x, y in amostra["cantos"]],
            "caixa_contida": caixa,
            "tipo_modelo_foco": modelo_foco["tipo"],
            # Os pontos e o modelo como saíram da Fase 2, intocados.
            "pontos_calibracao": [[x, y, z] for x, y, z in amostra["pontos_foco"]],
            "modelo_foco_inicial": amostra["modelo_foco"],
            # E como ficaram depois da re-ancoragem. Guardar os dois é o que
            # permite medir, rodada após rodada, quanto o foco anda por hora
            # de espera -- que é o número que dimensiona a busca grossa.
            "pontos_calibracao_reancorados": [[x, y, z] for x, y, z in pontos_foco],
            "modelo_foco_reancorado": modelo_foco,
            "reancoragem": registro_reancoragem,
            # Previstos com o modelo que a varredura de fato vai usar.
            "z_minimo_previsto": z_minimo,
            "z_maximo_previsto": z_maximo,
            "z_minimo_previsto_fase2": amostra["z_minimo"],
            "z_maximo_previsto_fase2": amostra["z_maximo"],
            "autofoco": config_autofoco,
            "timestamp_lote": timestamp,
        },
    )

    caminho_autofoco = os.path.join(diretorio, "autofoco.csv")

    with open(caminho_autofoco, "w", newline="", encoding="utf-8-sig") as arquivo:
        csv.writer(arquivo, delimiter=";").writerow(CABECALHO_AUTOFOCO)

    malha_x = malha["malha_x"]
    malha_y = malha["malha_y"]
    total = len(malha_x) * len(malha_y)

    contador = 0
    inicio = time.time()

    # Onde o tempo vai. As etapas não se sobrepõem: o "autofoco" mede o
    # checkpoint inteiro, que faz os PRÓPRIOS movimentos de Z e as próprias
    # fotos -- eles não passam pelos cronômetros de "mover_z" e "capturar",
    # que só envolvem as chamadas feitas aqui no laço.
    cronometro = Cronometro()

    # Quantos checkpoints terminaram em cada desfecho. É o que responde, no
    # fim, se o autofoco esteve trabalhando ou apenas recusando tudo -- e
    # essa é a diferença entre uma amostra em foco e uma amostra perdida.
    contagem_acoes = {}

    # As correções que o autofoco de fato aplicou. Elas medem quanto o foco
    # ANDOU entre dois checkpoints -- e é isso, e não a contagem de
    # descartes, que denuncia um INTERVALO_AUTOFOCO grande demais: com
    # todos os checkpoints funcionando perfeitamente, a amostra ainda sai
    # borrada se o foco escapar uma profundidade de campo entre eles.
    correcoes_aplicadas = []

    # Serpentina: desce uma coluna, sobe a seguinte. O (i, j) que sai daqui
    # é sempre o índice GEOMÉTRICO da grade -- é ele que nomeia o arquivo,
    # então o mosaico é montado igual, independente da ordem de visita.
    for i, j in ordem_serpentina(len(malha_x), len(malha_y)):
        n = malha_x[i]
        m = malha_y[j]

        contador += 1

        z_previsto = calcular_z(n, m, modelo_foco) + ajuste_z

        if not np.isfinite(z_previsto):
            raise ValueError(f"Z calculado inválido em X={n}, Y={m}: Z={z_previsto}")

        with cronometro.medir("mover_xy"):
            core.set_xy_position(xy_stage, n, m)
            core.wait_for_device(xy_stage)

        with cronometro.medir("mover_z"):
            core.set_position(z_stage, z_previsto)
            core.wait_for_device(z_stage)

        with cronometro.medir("capturar"):
            imagem = capturar_imagem(core, camera)

        z_final = z_previsto
        registro = None

        # O contador == 1 é o autofoco de abertura, com a faixa alargada.
        if config_autofoco and (contador == 1 or contador % INTERVALO_AUTOFOCO == 0):
            with cronometro.medir("autofoco"):
                imagem, z_final, modelo_foco, pontos_foco, ajuste_z, registro = checkpoint_autofoco(
                    core,
                    z_stage,
                    camera,
                    imagem,
                    n,
                    m,
                    z_previsto,
                    modelo_foco,
                    pontos_foco,
                    ajuste_z,
                    config_autofoco,
                    fator_faixa=FATOR_FAIXA_ABERTURA if contador == 1 else 1.0,
                    qtd_ancoras=qtd_ancoras,
                    estado=estado_foco,
                    # No primeiro checkpoint não existe correção anterior
                    # contra a qual julgar esta -- ver `abertura` no
                    # checkpoint_autofoco.
                    abertura=(contador == 1),
                )

        nome_arquivo = f"img_pos_X{i}_Y{j}.tif"

        with cronometro.medir("salvar"):
            imwrite(os.path.join(diretorio, nome_arquivo), imagem)

        if registro is not None:
            contagem_acoes[registro["acao"]] = contagem_acoes.get(registro["acao"], 0) + 1

            if registro["acao"] in ("modelo_reajustado", "offset"):
                correcoes_aplicadas.append(abs(registro["correcao"]))

            with open(caminho_autofoco, "a", newline="", encoding="utf-8-sig") as arquivo:
                csv.writer(arquivo, delimiter=";").writerow([
                    contador, i, j, f"{n:.2f}", f"{m:.2f}",
                    f"{z_previsto:.3f}", f"{z_final:.3f}",
                    f"{registro['textura']:.3f}",
                    f"{registro['z_medido']:.3f}",
                    f"{registro['correcao']:.3f}",
                    registro["confiavel"],
                    registro["acao"],
                    registro["pontos_modelo"],
                    registro["fotos"],
                    f"{registro['queda']:.4f}",
                    f"{registro['residuo_rms']:.3f}",
                ])

        if contador % 25 == 0 or contador == total:
            decorrido = time.time() - inicio
            restante = decorrido / contador * (total - contador)

            print(
                f"  {contador}/{total} imagens "
                f"({decorrido / 60:.0f} min decorridos, ~{restante / 60:.0f} min restantes)"
            )

    decorrido = time.time() - inicio
    cronometrado = sum(cronometro.total.values())

    if contagem_acoes:
        checkpoints = sum(contagem_acoes.values())
        aceitos = contagem_acoes.get("modelo_reajustado", 0) + contagem_acoes.get("offset", 0)

        print(f"\n  Autofoco (amostra {amostra['numero']}): {checkpoints} checkpoints")

        for acao, quantidade in sorted(contagem_acoes.items(), key=lambda par: -par[1]):
            print(f"    {acao:<24}{quantidade:>4}  ({quantidade / checkpoints:.0%})")

        if correcoes_aplicadas:
            correcao_tipica = float(np.median(correcoes_aplicadas))

            # A profundidade de campo, deduzida da faixa configurada -- o
            # parametros.py recomenda faixa ~ 2,5 x DOF.
            dof = config_autofoco["faixa_um"] / 2.5

            print(
                f"    correção típica          {correcao_tipica:>6.2f} µm "
                f"(profundidade de campo ~{dof:.1f} µm)"
            )

            # Se o foco anda mais que a DOF entre dois checkpoints, as
            # imagens do meio do intervalo saíram fora de foco -- mesmo com
            # todo checkpoint funcionando. O sintoma não aparece na
            # contagem de descartes, só aqui.
            if correcao_tipica > dof:
                print(
                    f"\n  [ATENÇÃO] entre um checkpoint e o seguinte o foco andou "
                    f"{correcao_tipica:.1f} µm,\n"
                    f"            mais que a profundidade de campo (~{dof:.1f} µm). Os "
                    "checkpoints\n"
                    "            funcionaram, mas as imagens ENTRE eles saíram fora de "
                    "foco.\n"
                    f"            Diminua o INTERVALO_AUTOFOCO (hoje {INTERVALO_AUTOFOCO}) "
                    "-- com a busca\n"
                    "            do centro para fora, cada checkpoint custa ~3 fotos, "
                    "então\n"
                    "            checar com o dobro da frequência é barato."
                )

        # O aviso existe porque uma varredura desassistida que se perde não
        # avisa sozinha: ela simplesmente grava horas de imagem borrada, e
        # isso só aparece quando o mosaico é montado. Aceitar menos da
        # metade dos checkpoints quer dizer que o foco andou mais do que o
        # sistema conseguiu acompanhar.
        if aceitos < checkpoints / 2:
            print(
                f"\n  [ATENÇÃO] só {aceitos} de {checkpoints} checkpoints corrigiram o\n"
                "            modelo. Esta amostra provavelmente saiu fora de foco em boa\n"
                "            parte da área. Veja a coluna Acao do autofoco.csv:\n"
                "              descartado_na_borda    -> a faixa está curta para o que o\n"
                "                                        foco andou; aumente autofoco_faixa_um\n"
                "                                        ou diminua INTERVALO_AUTOFOCO.\n"
                "              descartado_pico_chato  -> os campos não têm estrutura em que\n"
                "                                        medir foco; suba a textura_minima\n"
                "                                        para pular esses campos mais cedo.\n"
                "              descartado_outlier     -> a amostra tem muito detrito, ou o\n"
                "                                        foco saltou de verdade."
            )

    print(f"\n  Onde foi o tempo (amostra {amostra['numero']}):")
    print(cronometro.resumo(indentacao="    "))
    print(
        f"    total do laço {decorrido / 60:.1f} min, dos quais "
        f"{cronometrado / decorrido:.0%} nas etapas acima "
        f"({decorrido / total:.2f} s por imagem no fim das contas)"
    )

    return total, cronometro


print(f"\nIniciando. Arquivos em: {diretorio_lote}")

falhas = []
inicio_lote = time.time()

# Um cronômetro do lote inteiro, alimentado pelo de cada amostra. Serve
# para comparar RODADAS (mudou a objetiva? o passo? o disco?) -- a média de
# uma amostra só é ruidosa demais para isso.
cronometro_lote = Cronometro()
tempos_por_amostra = {}

try:
    for amostra in amostras:
        numero = amostra["numero"]

        print("\n" + "=" * 70)
        print(f"VARRENDO AMOSTRA {numero} de {qtd_amostras}")
        print("=" * 70)

        # Uma amostra que falha não pode levar as outras junto: numa rodada
        # de cinco, um erro na terceira desperdiçaria a quarta e a quinta
        # por nada. O KeyboardInterrupt NÃO é pego aqui de propósito -- ele
        # herda de BaseException, não de Exception, então o seu Ctrl+C
        # continua parando tudo, que é o que você quer dizer com ele.
        try:
            total, cronometro_amostra = varrer_amostra(amostra)

            cronometro_lote.somar(cronometro_amostra)
            tempos_por_amostra[numero] = cronometro_amostra.como_dicionario()

            print(f"[OK] Amostra {numero} concluída: {total} imagens em {amostra['diretorio']}")

        except Exception as erro:
            falhas.append((numero, erro))

            print(f"\n[FALHA] A amostra {numero} foi interrompida: {erro}")
            print("        Seguindo para a próxima amostra.\n")

finally:
    print("\nRetornando o estágio à posição inicial...")
    core.set_xy_position(xy_stage, x_origem, y_origem)
    core.wait_for_device(xy_stage)

duracao_lote = time.time() - inicio_lote

print("\n" + "=" * 70)
print(f"RODADA CONCLUÍDA em {duracao_lote / 3600:.1f} h")
print("=" * 70)

if cronometro_lote.total:
    print("\nOnde foi o tempo, na rodada inteira:")
    print(cronometro_lote.resumo())
    print(
        "\n  A etapa mais cara é por onde começa qualquer otimização de tempo.\n"
        "  'autofoco' inclui os movimentos de Z e as fotos do próprio checkpoint."
    )

    # Gravado, e não só impresso: comparar duas rodadas exige o número, e o
    # console some assim que a janela fecha.
    with open(os.path.join(diretorio_lote, "tempos.json"), "w", encoding="utf-8") as arquivo:
        json.dump(
            {
                "duracao_total_s": round(duracao_lote, 1),
                "total_imagens": total_imagens,
                "intervalo_autofoco": INTERVALO_AUTOFOCO,
                "objetiva": nome_objetiva,
                "autofoco": config_autofoco,
                "lote": cronometro_lote.como_dicionario(),
                "por_amostra": tempos_por_amostra,
            },
            arquivo,
            indent=2,
            ensure_ascii=False,
            default=float,
        )

if falhas:
    print(f"\n{len(falhas)} amostra(s) falharam:")

    for numero, erro in falhas:
        print(f"  Amostra {numero}: {erro}")

print(f"\nTudo em: {diretorio_lote}")
print("\nPara montar o mosaico de cada amostra:")

for amostra in amostras:
    if amostra["numero"] not in [numero for numero, _ in falhas]:
        print(f'    python montar_mosaico.py "{amostra["diretorio"]}"')

if config_autofoco is None:
    print(
        "\nO autofoco rodou DESLIGADO, então os autofoco.csv saíram vazios (só o\n"
        "cabeçalho) -- e é neles que a textura_minima seria medida.\n"
        "\n" 
        "Para sair desse impasse: ponha faixa e passo (do z-stack) e deixe\n"
        f'    OBJETIVAS["{nome_objetiva}"]["textura_minima"] = 0.0\n'
        "Com 0.0 o portão nunca fecha -- o autofoco roda em TODA posição de\n"
        "checkpoint, inclusive nos campos vazios, e a coluna Textura do CSV passa a\n"
        "mostrar o valor de cada um. O limiar de verdade é um número entre o que os\n"
        "campos vazios deram e o que os campos com estrutura deram."
    )
else:
    print(
        "\nPara calibrar a textura_minima desta objetiva, abra os autofoco.csv: a\n"
        "coluna Textura separa campo vazio de campo com estrutura.\n"
        "\n"
        "O que mais ler nesse arquivo:\n"
        "    Acao          por que cada checkpoint foi aceito ou descartado. Muito\n"
        "                  'descartado_na_borda' -> a faixa está curta para a deriva\n"
        "                  desta amostra; muito 'descartado_outlier' -> ou a amostra\n"
        "                  tem detrito demais, ou o limiar está apertado.\n"
        "    Fotos         o que cada checkpoint custou. Perto de 3 é o modelo\n"
        "                  acertando; perto do máximo é a busca atravessando a faixa.\n"
        "    Correcao      contra Contador mostra DERIVA (foco andando com o tempo);\n"
        "                  contra X e Y mostra erro do MODELO (a superfície não é a\n"
        "                  que ele supõe).\n"
        "    Residuo_RMS   se cresce ao longo da varredura, o modelo global (plano ou\n"
        "                  quadrática) não está dando conta da forma real da amostra."
    )
