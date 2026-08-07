"""Funções de calibração de foco e geração de malha de varredura,
compartilhadas entre deteccao.py e varredura_mosaico.py."""

import json
import math
import os

import cv2
import numpy as np

from capturar_imagem import capturar_imagem

# Limite de condicionamento aceitável para os ajustes de foco.
#
# O número de condição da matriz de projeto é o FATOR DE AMPLIFICAÇÃO do
# erro dos pontos de calibração no resultado: com cond = 1e6, um erro
# humano de 0,4 µm ao focar um ponto vira até 4e5 µm no Z previsto.
#
# O que estraga o condicionamento é a falta de NÍVEIS -- a quantidade de
# valores distintos que cada coordenada assume no conjunto de pontos --, e
# não a regularidade do arranjo. Para estimar grau d num eixo são precisos
# d+1 níveis nesse eixo: o plano (grau 1) exige 2, a quadrática (grau 2)
# exige 3. O motivo é direto: com só 2 níveis, ux² pode ser interpolado
# exatamente por uma reta em ux, então a coluna ux² vira combinação linear
# das colunas ux e 1s e a matriz fica SINGULAR -- não apenas mal
# condicionada. É o caso dos quatro cantos de um retângulo (2 níveis em
# cada eixo) e o de um grid 5x2 (2 níveis em Y): ambos medem cond ~1e16.
#
# Um teste de posto não substitui esta verificação. Ele pega o caso
# exatamente singular, mas não o "quase", que é o que aparece na prática
# quando os pontos estão só PERTO de degenerados -- ali o posto dá 6,
# cheio, e o ajuste passa devolvendo lixo. Para referência: um grid 3x3
# regular dá cond ~4, e 10 pontos sorteados ao acaso, ~11.
LIMITE_CONDICIONAMENTO = 1e4


def _matriz_projeto(xs, ys, tipo):
    """Monta a matriz de projeto do ajuste de foco para as coordenadas
    dadas e devolve (matriz, x0, y0, escala).

    Vive numa função própria porque o condicionamento precisa ser medido
    em DOIS momentos: no ajuste, com os pontos que o usuário de fato
    focou, e ANTES da coleta, só com as coordenadas sugeridas -- é isso
    que permite avisar sobre uma geometria ruim enquanto ainda dá tempo de
    mudar, em vez de recusar tudo depois de dez focagens manuais. Como o
    condicionamento não depende de Z, os dois cálculos são o mesmo; se
    cada lugar montasse a sua própria matriz, o aviso prévio e a recusa
    final poderiam discordar."""

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    x0 = float(np.mean(xs))
    y0 = float(np.mean(ys))

    dx = xs - x0
    dy = ys - y0

    if tipo != "quadratica":
        # O plano ajusta dz (também centrado) contra dx e dy, por isso não
        # há coluna de 1s aqui: o z0 já é a média dos Z.
        return np.column_stack([dx, dy]), x0, y0, 1.0

    # Normaliza as coordenadas ANTES de montar a matriz, para que as
    # colunas fiquem na mesma ordem de grandeza: sem isso, numa área de
    # ~500 µm, dx fica nas centenas e dx² nas centenas de milhares. Isso
    # melhora o condicionamento cerca de 25x (medido: 2,5e8 -> 9,8e6).
    # Sozinho NÃO resolve: o que domina é a geometria dos pontos, não a
    # escala das unidades -- ver LIMITE_CONDICIONAMENTO.
    escala = float(max(np.abs(dx).max(), np.abs(dy).max()))

    if escala == 0:
        raise ValueError(
            "ERRO: todos os pontos de calibração estão na mesma posição XY."
        )

    ux = dx / escala
    uy = dy / escala

    matriz = np.column_stack([ux, uy, ux * uy, ux**2, uy**2, np.ones(len(xs))])

    return matriz, x0, y0, escala


def _condicionamento(xs, ys, tipo):
    """Fator de amplificação do erro de foco para uma dada geometria de
    pontos. Depende só de X e Y -- é por isso que dá para avaliar a
    qualidade de um conjunto de pontos antes de focar qualquer um deles."""

    matriz, _, _, _ = _matriz_projeto(xs, ys, tipo)

    return float(np.linalg.cond(matriz))


def calibrar_plano_foco(pontos, verboso=True):
    """Ajusta um plano de foco da forma:
        Z = Z0 + ax*(X - X0) + ay*(Y - Y0).

    verboso=False silencia o resumo impresso -- usado no re-ajuste
    automático durante a varredura, que roda dezenas de vezes e encheria
    o console de tabelas."""

    pontos = np.array(pontos, dtype=float)

    if len(pontos) < 3:
        raise ValueError(
            "ERRO: São necessários pelo menos 3 pontos para ajustar o plano de foco."
        )

    xs = pontos[:, 0]
    ys = pontos[:, 1]
    zs = pontos[:, 2]

    matriz, x0, y0, _ = _matriz_projeto(xs, ys, "plano")

    # Centro em Z dos pontos de calibração. O ajuste é feito sobre dz,
    # centrado, e por isso a matriz do plano não tem coluna de 1s.
    z0 = float(np.mean(zs))
    dz = zs - z0

    # Pontos alinhados (ou quase) deixam o plano mal determinado. O número
    # de condição pega tanto o caso exato quanto o "quase", que é o que
    # acontece na prática -- um teste de posto só pegaria o exato.
    condicionamento = np.linalg.cond(matriz)

    if not np.isfinite(condicionamento) or condicionamento > LIMITE_CONDICIONAMENTO:
        raise ValueError(
            "ERRO: Os pontos de calibração estão alinhados ou quase alinhados "
            f"(condicionamento {condicionamento:.1e}, limite {LIMITE_CONDICIONAMENTO:.0e}). "
            "Escolha pelo menos 3 pontos bem espalhados pela amostra, formando "
            "um triângulo largo em vez de uma linha."
        )

    coeficientes, _, _, _ = np.linalg.lstsq(matriz, dz, rcond=None)

    ax = coeficientes[0]
    ay = coeficientes[1]

    plano = {
        "tipo": "plano",
        "x0": x0,
        "y0": y0,
        "z0": z0,
        # O plano não precisa de normalização (dx e dy já estão na mesma
        # ordem de grandeza), mas o campo existe para o calcular_z poder
        # tratar os dois modelos do mesmo jeito.
        "escala": 1.0,
        "ax": ax,
        "ay": ay,
    }

    if verboso:
        _imprimir_modelo_foco(plano, pontos)

    return plano


def calibrar_superficie_foco(pontos, verboso=True):
    """Ajusta uma superfície quadrática de foco da forma:
        Z = Z0 + ax*dx + ay*dy + axy*dx*dy + axx*dx² + ayy*dy²
    onde dx = X - X0, dy = Y - Y0.

    verboso=False silencia o resumo impresso (ver calibrar_plano_foco)."""

    pontos = np.array(pontos, dtype=float)

    if len(pontos) < 6:
        raise ValueError(
            "ERRO: A superfície quadrática tem 6 coeficientes e precisa de pelo "
            "menos 6 pontos. Na prática, use mais -- com exatamente 6 o ajuste "
            "passa exatamente por eles e reproduz o erro de cada um sem nenhuma "
            "média para amortecê-lo."
        )

    xs = pontos[:, 0]
    ys = pontos[:, 1]
    zs = pontos[:, 2]

    matriz_teste, x0, y0, escala = _matriz_projeto(xs, ys, "quadratica")

    condicionamento = np.linalg.cond(matriz_teste)

    if not np.isfinite(condicionamento) or condicionamento > LIMITE_CONDICIONAMENTO:
        raise ValueError(
            "ERRO: Os pontos de calibração estão mal distribuídos para a "
            f"superfície quadrática (condicionamento {condicionamento:.1e}, "
            f"limite {LIMITE_CONDICIONAMENTO:.0e}). Nesse estado, o erro ao "
            "focar cada ponto seria amplificado por esse fator no Z previsto. "
            "A causa quase sempre é falta de níveis: a quadrática precisa de "
            "pelo menos 3 valores DISTINTOS de X e 3 de Y entre os pontos. "
            "Quatro cantos, uma fileira ou uma diagonal não atendem a isso, "
            "por mais pontos que se acrescente sobre o mesmo padrão."
        )

    coeficientes, _, _, _ = np.linalg.lstsq(matriz_teste, zs, rcond=None)

    ax, ay, axy, axx, ayy, z0 = coeficientes

    superficie = {
        "tipo": "quadratica",
        "x0": x0,
        "y0": y0,
        "z0": z0,
        "escala": escala,
        "ax": ax,
        "ay": ay,
        "axy": axy,
        "axx": axx,
        "ayy": ayy,
    }

    if verboso:
        _imprimir_modelo_foco(superficie, pontos)

    return superficie


def reajustar_modelo_foco(pontos, tipo):
    """Re-ajusta o modelo de foco com todos os pontos acumulados até agora:
    os coletados à mão no início mais os medidos pelo autofoco durante a
    varredura.

    É isso que faz o modelo melhorar sozinho ao longo da varredura, em vez
    de depender de um offset escalar único -- um escalar aprendido numa
    posição só vale perto dela, e é justamente por isso que o foco se perde
    depois de uma translação grande.

    Devolve None (em vez de levantar erro) se o ajuste não for possível:
    poucos pontos, ou pontos ainda alinhados -- o que acontece de verdade
    enquanto a varredura não saiu da primeira coluna, já que todos os
    pontos medidos têm o mesmo X. Quem chama decide o que fazer; derrubar
    a varredura por causa de um refinamento que falhou seria péssimo
    negócio depois de horas de microscópio."""

    try:
        if tipo == "quadratica":
            return calibrar_superficie_foco(pontos, verboso=False)

        return calibrar_plano_foco(pontos, verboso=False)

    except (ValueError, np.linalg.LinAlgError):
        return None


def calcular_z(x, y, modelo_foco):

    # A escala é a mesma usada no ajuste (1.0 para o plano). Sem dividir
    # aqui também, os coeficientes da superfície quadrática seriam aplicados
    # a coordenadas de outra ordem de grandeza e o Z sairia absurdo.
    escala = modelo_foco.get("escala", 1.0)

    dx = (x - modelo_foco["x0"]) / escala
    dy = (y - modelo_foco["y0"]) / escala

    z = modelo_foco["z0"] + modelo_foco["ax"] * dx + modelo_foco["ay"] * dy

    if modelo_foco["tipo"] == "quadratica":
        z += (
            modelo_foco["axy"] * dx * dy
            + modelo_foco["axx"] * dx**2
            + modelo_foco["ayy"] * dy**2
        )

    return z


def _imprimir_modelo_foco(modelo_foco, pontos):
    """Imprime o resumo dos coeficientes do modelo de foco ajustado e o
    teste nos próprios pontos de calibração (Z real x Z previsto x erro).
    Usado tanto pelo ajuste de plano quanto pelo de superfície."""

    if modelo_foco["tipo"] == "quadratica":
        print("\n--- SUPERFÍCIE DE FOCO AJUSTADA ---")
    else:
        print("\n--- PLANO DE FOCO AJUSTADO ---")

    # Os coeficientes são ajustados em coordenadas normalizadas (ver
    # calibrar_superficie_foco). Aqui eles são convertidos de volta para
    # µm para que os números impressos tenham significado físico direto:
    # dZ/dX em µm de foco por µm de deslocamento.
    escala = modelo_foco.get("escala", 1.0)

    print(f"x0 = {modelo_foco['x0']:.3f}")
    print(f"y0 = {modelo_foco['y0']:.3f}")
    print(f"z0 = {modelo_foco['z0']:.3f}")
    print(f"dZ/dX = {modelo_foco['ax'] / escala:.6f}")
    print(f"dZ/dY = {modelo_foco['ay'] / escala:.6f}")

    if modelo_foco["tipo"] == "quadratica":
        print(f"dZ/dXdY = {modelo_foco['axy'] / escala**2:.9f}")
        print(f"dZ/dX² = {modelo_foco['axx'] / escala**2:.9f}")
        print(f"dZ/dY² = {modelo_foco['ayy'] / escala**2:.9f}")

    print("\n--- TESTE NOS PONTOS DE CALIBRAÇÃO ---")
    for k, (x, y, z_real) in enumerate(pontos, start=1):
        z_previsto = calcular_z(x, y, modelo_foco)
        erro = z_previsto - z_real

        print(
            f"Ponto {k}: "
            f"Z real = {z_real:.3f}, "
            f"Z calculado = {z_previsto:.3f}, "
            f"erro = {erro:.3f}"
        )


def _melhor_grid(qtd, minimo_niveis):
    """Escolhe (nx, ny): o maior grid com nx*ny <= qtd em que ambos os
    eixos tenham pelo menos minimo_niveis níveis. Empates são resolvidos
    pelo arranjo mais próximo de quadrado.

    Por que o maior grid que CABE, e não o menor que cobre: sobra é fácil
    de resolver -- os pontos excedentes vão para os centros das células e
    não mexem nos níveis existentes. Falta obrigaria a remover pontos do
    grid, e remover pode derrubar um nível e tornar a matriz singular,
    que é exatamente o que se está tentando evitar.

    Devolve None quando nenhum grid cabe. Isso só acontece no plano com 3
    pontos, já que o menor grid possível ali (2x2) já são 4."""

    melhor = None

    for nx in range(minimo_niveis, qtd + 1):
        for ny in range(minimo_niveis, qtd + 1):
            if nx * ny > qtd:
                continue

            # Critério, nesta ordem: aproveitar o máximo de pontos dentro
            # do grid e, entre os empates, equilibrar os níveis entre os
            # eixos. Um 3x7 mediria a curvatura em Y com sete níveis e a
            # de X com o mínimo de três, gastando pontos à toa.
            chave = (nx * ny, -abs(nx - ny))

            if melhor is None or chave > melhor[0]:
                melhor = (chave, (nx, ny))

    if melhor is None:
        return None

    return melhor[1]


def _ordenar_serpentina(pontos):
    """Reordena os pontos em serpentina -- colunas de X, com o sentido de
    Y invertido a cada coluna -- para o estágio não atravessar a amostra
    de ponta a ponta entre um ponto e o seguinte."""

    colunas = {}

    for x, y in pontos:
        # O arredondamento serve só para agrupar: pontos gerados na mesma
        # coluna podem diferir no último bit por causa do linspace.
        colunas.setdefault(round(x, 6), []).append((x, y))

    ordenados = []

    for indice, chave in enumerate(sorted(colunas)):
        descendo = indice % 2 == 1
        ordenados.extend(sorted(colunas[chave], key=lambda ponto: ponto[1], reverse=descendo))

    return ordenados


def sugerir_pontos_calibracao(malha, qtd, tipo):
    """Sugere qtd coordenadas (X, Y) bem distribuídas pela área que a
    varredura vai de fato percorrer.

    A ideia é tirar do usuário a escolha da GEOMETRIA -- onde o erro
    humano é sistemático e caro, já que os padrões intuitivos (quatro
    cantos, uma fileira, a diagonal) são justamente os singulares -- e
    deixar com ele só o que exige o olho no ocular: achar o foco. Como o
    condicionamento não depende de Z, o arranjo inteiro pode ser conferido
    aqui, antes de o usuário focar o primeiro ponto.

    Devolve um dicionário com os pontos já na ordem de visita, a descrição
    do arranjo, o condicionamento previsto e o espaçamento do grid -- este
    último serve para dizer ao usuário quanto ele pode se afastar de um
    alvo sem estragar o ajuste."""

    malha_x = malha["malha_x"]
    malha_y = malha["malha_y"]

    # Extremos da MALHA, e não o ponto inicial/final que o usuário marcou:
    # o modelo de foco precisa ser bom onde as imagens serão tiradas, e o
    # arredondamento para cima do número de passos faz a malha extrapolar
    # um pouco a área marcada. Usar min/max também dispensa saber em que
    # sentido a varredura corre.
    x_min, x_max = min(malha_x), max(malha_x)
    y_min, y_max = min(malha_y), max(malha_y)

    # Varredura de uma coluna (ou de uma linha) só: sem extensão naquele
    # eixo todos os pontos cairiam no mesmo X, e a matriz seria singular
    # por falta de níveis. Abre-se a faixa de um passo, que é o pedaço da
    # amostra que aquele tile de fato cobre.
    if x_max == x_min:
        x_min -= malha["passo_x"] / 2
        x_max += malha["passo_x"] / 2

    if y_max == y_min:
        y_min -= malha["passo_y"] / 2
        y_max += malha["passo_y"] / 2

    # Grau 1 precisa de 2 níveis por eixo, grau 2 precisa de 3.
    # Ver LIMITE_CONDICIONAMENTO.
    minimo_niveis = 3 if tipo == "quadratica" else 2

    grid = _melhor_grid(qtd, minimo_niveis)

    if grid is None:
        # Só o plano com 3 pontos chega aqui, e para ele o triângulo largo
        # é o arranjo ótimo (cond ~1,15 numa área quadrada).
        pontos = [
            (x_min, y_min),
            (x_max, y_min),
            ((x_min + x_max) / 2, y_max),
        ]
        descricao = "triângulo largo"
        espacamento = min(x_max - x_min, y_max - y_min) / 2

    else:
        nx, ny = grid

        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)

        pontos = [(float(x), float(y)) for x in xs for y in ys]
        descricao = f"grid {nx}x{ny}"

        faltam = qtd - nx * ny

        if faltam:
            # Os pontos que sobram vão para os centros das células: eles
            # acrescentam níveis intermediários em vez de mexer nos que já
            # existem, então nunca pioram o condicionamento.
            centros = [
                (float((xs[i] + xs[i + 1]) / 2), float((ys[j] + ys[j + 1]) / 2), i + j)
                for i in range(nx - 1)
                for j in range(ny - 1)
            ]

            # Ordem de tabuleiro: pegando primeiro as células de paridade
            # par, os poucos centros usados saem diagonalmente opostos em
            # vez de amontoados num canto.
            centros.sort(key=lambda centro: (centro[2] % 2, centro[2]))

            pontos += [(x, y) for x, y, _ in centros[:faltam]]
            descricao += f" + {faltam} centro(s) de célula"

        espacamento = min((x_max - x_min) / (nx - 1), (y_max - y_min) / (ny - 1))

    pontos = _ordenar_serpentina(pontos)

    return {
        "pontos": pontos,
        "descricao": descricao,
        "condicionamento": _condicionamento(
            [ponto[0] for ponto in pontos], [ponto[1] for ponto in pontos], tipo
        ),
        "espacamento": espacamento,
    }


def coletar_pontos_calibracao(core, xy_stage, z_stage, minimo_pontos, malha=None, tipo="plano"):
    """Pede a quantidade de pontos de calibração e depois coleta cada um.

    Com malha=None mantém o modo manual: o usuário escolhe sozinho onde
    colocar cada ponto. Recebendo a malha, o script passa a SUGERIR as
    posições e a levar o estágio até elas em XY -- ao usuário resta só
    ajustar o foco.

    O script nunca toca no Z. Antes da calibração ele não sabe onde está a
    superfície da amostra, e descer a objetiva às cegas arrisca bater
    nela; a partir de três pontos ele passa a IMPRIMIR uma estimativa de
    Z, que é informação, não movimento.

    O alvo sugerido é um alvo, não uma obrigação: caindo num pedaço vazio
    do substrato, o usuário move até onde houver estrutura focável. O que
    entra na lista é sempre a posição REAL lida do estágio, nunca a
    sugerida -- gravar a teórica introduziria um erro em X e Y que o
    ajuste depois interpretaria como variação de foco.

    Retorna a lista de pontos (X, Y, Z) coletados."""

    while True:
        entrada = input(f"Quantos pontos serão utilizados para ajustar a varredura? (Mínimo {minimo_pontos}): ")

        try:
            qtd_pontos_calibracao = int(entrada)

            if qtd_pontos_calibracao < minimo_pontos:
                print(
                    f"[Aviso] Você digitou {qtd_pontos_calibracao}. "
                    f"É necessário pelo menos {minimo_pontos} pontos para calibrar o estágio. "
                    "Tente novamente.\n"
                )
            else:
                print(f"[Sucesso] O sistema será calibrado utilizando {qtd_pontos_calibracao} pontos.\n")
                break

        except ValueError:
            print("[Erro] Entrada inválida! Por favor, digite apenas números inteiros. Exemplo: 3, 4, 5.\n")

    sugestao = None

    if malha is not None:
        sugestao = sugerir_pontos_calibracao(malha, qtd_pontos_calibracao, tipo)

        print(
            f"Arranjo sugerido: {sugestao['descricao']} -- condicionamento previsto "
            f"{sugestao['condicionamento']:.1f} (limite {LIMITE_CONDICIONAMENTO:.0e})."
        )

        if sugestao["condicionamento"] > LIMITE_CONDICIONAMENTO:
            # Nenhum rearranjo salva este caso: o condicionamento também
            # depende da razão de aspecto da ÁREA, e uma área muito
            # alongada estoura o limite mesmo com os pontos ideais.
            print(
                "[Aviso] Nem o melhor arranjo possível cabe no limite nesta área --\n"
                "        ela é alongada demais para o ajuste escolhido. O problema é a\n"
                "        forma da área, não a posição dos pontos, e mudar os pontos não\n"
                "        resolve. Considere usar o plano em vez da quadrática, ou varrer\n"
                "        uma região menos alongada."
            )
        elif sugestao["condicionamento"] > LIMITE_CONDICIONAMENTO / 100:
            print(
                "[Aviso] Condicionamento previsto alto (área alongada). Passa no limite,\n"
                "        mas o erro ao focar cada ponto será amplificado no Z previsto."
            )

        print(
            f"\nO estágio será levado até cada ponto sugerido. Se algum cair num pedaço\n"
            f"sem nada em que focar, mova livremente até uma região próxima que tenha\n"
            f"estrutura: o que vale é onde você parar. Desvios de até uns\n"
            f"{sugestao['espacamento']:.0f} µm não atrapalham o ajuste."
        )

    pontos_coletados = []

    for p in range(qtd_pontos_calibracao):
        alvo_x = alvo_y = None

        if sugestao is None:
            input(f"\nMova o estágio para o Ponto de Calibração {p + 1}, AJUSTE O FOCO, e pressione ENTER...")

        else:
            alvo_x, alvo_y = sugestao["pontos"][p]

            print(f"\n--- Ponto {p + 1} de {qtd_pontos_calibracao} ---")
            print(f"Levando o estágio para X={alvo_x:.2f}, Y={alvo_y:.2f}...")

            core.set_xy_position(xy_stage, alvo_x, alvo_y)
            core.wait_for_device(xy_stage)

            # Com 3 pontos já dá para ajustar um plano provisório e dizer
            # em que Z o foco provavelmente está. É só um número impresso:
            # quem move o Z continua sendo o usuário.
            if len(pontos_coletados) >= 3:
                modelo_previo = reajustar_modelo_foco(pontos_coletados, "plano")

                if modelo_previo is not None:
                    print(
                        f"Z estimado pelos {len(pontos_coletados)} pontos anteriores: "
                        f"{calcular_z(alvo_x, alvo_y, modelo_previo):.3f} "
                        f"(estimativa; o script não mexe no Z)"
                    )

            input("AJUSTE O FOCO e pressione ENTER...")

        x_atual = core.get_x_position(xy_stage)
        y_atual = core.get_y_position(xy_stage)
        z_atual = core.get_position(z_stage)

        pontos_coletados.append((x_atual, y_atual, z_atual))

        print(f"-> Ponto {p + 1} salvo: X={x_atual:.1f}, Y={y_atual:.1f}, Z={z_atual:.2f}")

        if alvo_x is not None:
            desvio = math.hypot(x_atual - alvo_x, y_atual - alvo_y)

            if desvio > sugestao["espacamento"]:
                print(
                    f"   [Aviso] desvio de {desvio:.0f} µm do alvo, maior que o espaçamento "
                    f"do arranjo ({sugestao['espacamento']:.0f} µm). Registrado assim mesmo; "
                    "a conferência abaixo dirá se a geometria ainda serve."
                )

    if sugestao is not None:
        # O condicionamento previsto valia para os alvos. Este é o dos
        # pontos que o usuário de fato focou -- é ele que vale, e vê-lo
        # aqui evita descobrir um problema só quando o ajuste recusar.
        condicionamento = _condicionamento(
            [ponto[0] for ponto in pontos_coletados],
            [ponto[1] for ponto in pontos_coletados],
            tipo,
        )

        print(
            f"\nCondicionamento dos pontos realmente coletados: {condicionamento:.1f} "
            f"(previsto: {sugestao['condicionamento']:.1f}, limite {LIMITE_CONDICIONAMENTO:.0e})."
        )

    return pontos_coletados


def gerar_malha_varredura(core, xy_stage, fov_x, fov_y, sobreposicao=0.0):
    """Pede o ponto inicial e final da área a varrer e gera a malha de
    posições (X, Y) cobrindo essa área.

    O espaçamento entre posições é o campo de visão MENOS a sobreposição:
    com sobreposicao=0.1 o estágio avança 90% do campo a cada passo, e
    tiles vizinhos compartilham uma faixa de 10%. Isso evita que um flake
    em cima da divisa seja cortado em dois pedaços medidos separadamente.

    Retorna um dicionário com a malha, o passo usado e os dados que a
    geraram. O passo é devolvido porque quem monta o mosaico depois
    precisa dele: sem saber o passo, o montador encaixaria os tiles como
    se fossem adjacentes e duplicaria a faixa sobreposta."""

    print("--- CONFIGURAÇÃO DE MAPEAMENTO ---")

    input("1. Mova o estágio para o PONTO INICIAL (ex: canto superior esquerdo da amostra) e pressione ENTER...")
    x_inicial = core.get_x_position(xy_stage)
    y_inicial = core.get_y_position(xy_stage)

    print(f"Ponto Inicial registrado: X={x_inicial:.2f}, Y={y_inicial:.2f}")

    input("2. Mova o estágio para o PONTO FINAL (ex: canto inferior direito da amostra) e pressione ENTER...")
    x_final = core.get_x_position(xy_stage)
    y_final = core.get_y_position(xy_stage)

    print(f"Ponto Final registrado: X={x_final:.2f}, Y={y_final:.2f}")

    direcao_x = 1 if x_final > x_inicial else -1
    direcao_y = 1 if y_final > y_inicial else -1

    dist_x = abs(x_final - x_inicial)
    dist_y = abs(y_final - y_inicial)

    if not 0.0 <= sobreposicao < 1.0:
        raise ValueError(
            f"ERRO: sobreposicao deve estar em [0, 1), recebido {sobreposicao}."
        )

    passo_x = fov_x * (1.0 - sobreposicao)
    passo_y = fov_y * (1.0 - sobreposicao)

    qtd_passos_x = math.ceil(dist_x / passo_x)
    qtd_passos_y = math.ceil(dist_y / passo_y)

    malha_x = [x_inicial + (i * passo_x * direcao_x) for i in range(qtd_passos_x)]
    malha_y = [y_inicial + (j * passo_y * direcao_y) for j in range(qtd_passos_y)]

    if sobreposicao > 0:
        print(
            f"\nSobreposição de {sobreposicao:.0%}: passo de "
            f"{passo_x:.2f} x {passo_y:.2f} µm num campo de {fov_x:.2f} x {fov_y:.2f} µm "
            f"(faixa compartilhada de {fov_x - passo_x:.2f} x {fov_y - passo_y:.2f} µm)."
        )

    return {
        "malha_x": malha_x,
        "malha_y": malha_y,
        "x_inicial": x_inicial,
        "y_inicial": y_inicial,
        "dist_x": dist_x,
        "dist_y": dist_y,
        "qtd_passos_x": qtd_passos_x,
        "qtd_passos_y": qtd_passos_y,
        "passo_x": passo_x,
        "passo_y": passo_y,
        "sobreposicao": sobreposicao,
    }


def salvar_metadados_varredura(diretorio, malha, fov_x, fov_y):
    """Grava na pasta da varredura o que o montar_mosaico.py precisa saber
    para remontar a amostra na escala certa.

    Por que num arquivo, em vez de o montador simplesmente ler a constante
    SOBREPOSICAO do parametros.py: se você mudar essa constante entre rodar
    a varredura e montar o mosaico, o montador usaria o valor NOVO em
    imagens tiradas com o valor ANTIGO, e o mosaico sairia desalinhado sem
    nenhum aviso. Gravado aqui junto das imagens, o arquivo descreve o que
    aquela varredura de fato fez, e não o que o parametros.py diz hoje."""

    metadados = {
        "fov_x": fov_x,
        "fov_y": fov_y,
        "passo_x": malha["passo_x"],
        "passo_y": malha["passo_y"],
        "sobreposicao": malha["sobreposicao"],
        "grid_x": malha["qtd_passos_x"],
        "grid_y": malha["qtd_passos_y"],
    }

    caminho = os.path.join(diretorio, "varredura.json")

    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(metadados, arquivo, indent=2, ensure_ascii=False)

    return caminho


def medir_nitidez(imagem):
    """Mede o quão nítida (em foco) uma imagem está, usando a variância do
    Laplaciano -- quanto maior o valor, mais nítida a imagem está.
    É a métrica clássica de autofoco por imagem em microscopia.

    ATENÇÃO: o valor só é comparável entre imagens do MESMO campo de visão.
    Ele depende fortemente do conteúdo: um campo vazio em foco perfeito
    pontua parecido com um campo cheio de flakes fora de foco, porque em
    campo vazio o Laplaciano mede o ruído do sensor, não o foco. Por isso
    existe a medir_textura -- para decidir se dá pra medir foco aqui."""

    cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY) if imagem.ndim == 3 else imagem
    return cv2.Laplacian(cinza, cv2.CV_64F).var()


def medir_textura(imagem):
    """Mede quanta estrutura REAL existe na imagem -- serve para decidir se
    faz sentido rodar autofoco neste campo.

    É um filtro passa-banda: alisar com sigma pequeno derruba o ruído do
    sensor (alta frequência) e subtrair uma versão muito borrada derruba a
    vinhetagem e o gradiente de iluminação (baixa frequência). Sobra só a
    estrutura de escala intermediária, que é o que os flakes são.

    Ao contrário da medir_nitidez, esse valor quase não muda com o foco:
    um campo com flakes continua pontuando alto mesmo bem desfocado. É
    exatamente o que se quer num portão de decisão -- ele não pode
    bloquear o autofoco justo quando a imagem está fora de foco."""

    cinza = imagem if imagem.ndim == 2 else cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    cinza = cinza.astype(np.float32)

    suave = cv2.GaussianBlur(cinza, (0, 0), 2.0)
    fundo = cv2.GaussianBlur(cinza, (0, 0), 32.0)

    return float((suave - fundo).std())


def autofocar(core, z_stage, camera, z_estimado, faixa_um, passo_um):
    """Faz uma pequena varredura fina em Z ao redor de z_estimado (o Z que o
    modelo de foco previu) e fica com a posição mais nítida encontrada,
    varrendo +-faixa_um em passos de passo_um.

    Retorna (melhor_z, melhor_imagem, medicao_confiavel). A imagem já vem
    capturada no Z escolhido, pra não precisar tirar outra foto depois.

    Sobre medicao_confiavel: se o Z vencedor caiu numa das PONTAS da faixa
    varrida, o foco de verdade provavelmente está fora do alcance da busca
    -- o que se encontrou foi só o melhor de um intervalo todo ruim. Isso é
    uma medição falhada, não um resultado, e quem chama não deve usá-la nem
    como ajuste nem como ponto de calibração."""

    deltas = np.arange(-faixa_um, faixa_um + passo_um, passo_um)

    melhor_z = z_estimado
    melhor_nitidez = -1.0
    melhor_imagem = None
    melhor_indice = -1

    for indice, delta in enumerate(deltas):
        z_teste = z_estimado + delta

        core.set_position(z_stage, z_teste)
        core.wait_for_device(z_stage)

        imagem = capturar_imagem(core, camera)
        nitidez = medir_nitidez(imagem)

        if nitidez > melhor_nitidez:
            melhor_nitidez = nitidez
            melhor_z = z_teste
            melhor_imagem = imagem
            melhor_indice = indice

    # Deixa o estágio de fato parado na melhor posição encontrada.
    core.set_position(z_stage, melhor_z)
    core.wait_for_device(z_stage)

    medicao_confiavel = 0 < melhor_indice < len(deltas) - 1

    return melhor_z, melhor_imagem, medicao_confiavel
