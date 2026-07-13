"""Funções de calibração de foco e geração de malha de varredura,
compartilhadas entre deteccao.py e varredura_mosaico.py."""

import math

import numpy as np


def calibrar_plano_foco(pontos):
    """Ajusta um plano de foco da forma:
        Z = Z0 + ax*(X - X0) + ay*(Y - Y0)."""

    pontos = np.array(pontos, dtype=float)

    xs = pontos[:, 0]
    ys = pontos[:, 1]
    zs = pontos[:, 2]

    # Centro dos pontos de calibração
    x0 = np.mean(xs)
    y0 = np.mean(ys)
    z0 = np.mean(zs)

    dx = xs - x0
    dy = ys - y0
    dz = zs - z0

    # Teste para evitar pontos alinhados.
    # Se os pontos estiverem alinhados, o plano de foco não fica bem determinado.
    matriz_teste = np.column_stack([dx, dy, np.ones(len(pontos))])
    posto = np.linalg.matrix_rank(matriz_teste)

    if posto < 3:
        raise ValueError(
            "ERRO: Os pontos de calibração estão alinhados ou quase alinhados. "
            "Escolha pelo menos 3 pontos bem espalhados pela amostra."
        )

    matriz = np.column_stack([dx, dy])

    coeficientes, _, _, _ = np.linalg.lstsq(matriz, dz, rcond=None)

    ax = coeficientes[0]
    ay = coeficientes[1]

    plano = {
        "tipo": "plano",
        "x0": x0,
        "y0": y0,
        "z0": z0,
        "ax": ax,
        "ay": ay,
    }

    _imprimir_modelo_foco(plano, pontos)

    return plano


def calibrar_superficie_foco(pontos):
    """Ajusta uma superfície quadrática de foco da forma:
        Z = Z0 + ax*dx + ay*dy + axy*dx*dy + axx*dx² + ayy*dy²
    onde dx = X - X0, dy = Y - Y0."""

    pontos = np.array(pontos, dtype=float)

    xs = pontos[:, 0]
    ys = pontos[:, 1]
    zs = pontos[:, 2]

    x0 = np.mean(xs)
    y0 = np.mean(ys)

    dx = xs - x0
    dy = ys - y0

    matriz_teste = np.column_stack([dx, dy, dx * dy, dx**2, dy**2, np.ones(len(pontos))])
    posto = np.linalg.matrix_rank(matriz_teste)

    if posto < 6:
        raise ValueError(
            "ERRO: Os pontos de calibração não são suficientes ou estão mal "
            "distribuídos para ajustar a superfície quadrática. "
            "Escolha pelo menos 6 pontos bem espalhados pela amostra."
        )

    coeficientes, _, _, _ = np.linalg.lstsq(matriz_teste, zs, rcond=None)

    ax, ay, axy, axx, ayy, z0 = coeficientes

    superficie = {
        "tipo": "quadratica",
        "x0": x0,
        "y0": y0,
        "z0": z0,
        "ax": ax,
        "ay": ay,
        "axy": axy,
        "axx": axx,
        "ayy": ayy,
    }

    _imprimir_modelo_foco(superficie, pontos)

    return superficie


def calcular_z(x, y, modelo_foco):

    dx = x - modelo_foco["x0"]
    dy = y - modelo_foco["y0"]

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

    print(f"x0 = {modelo_foco['x0']:.3f}")
    print(f"y0 = {modelo_foco['y0']:.3f}")
    print(f"z0 = {modelo_foco['z0']:.3f}")
    print(f"dZ/dX = {modelo_foco['ax']:.6f}")
    print(f"dZ/dY = {modelo_foco['ay']:.6f}")

    if modelo_foco["tipo"] == "quadratica":
        print(f"dZ/dXdY = {modelo_foco['axy']:.6f}")
        print(f"dZ/dX² = {modelo_foco['axx']:.6f}")
        print(f"dZ/dY² = {modelo_foco['ayy']:.6f}")

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


def coletar_pontos_calibracao(core, xy_stage, z_stage, minimo_pontos):
    """Pede a quantidade de pontos de calibração e depois coleta cada um
    deles movendo o estágio manualmente. Retorna a lista de pontos
    (X, Y, Z) coletados."""

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

    pontos_coletados = []

    for p in range(qtd_pontos_calibracao):
        input(f"\nMova o estágio para o Ponto de Calibração {p + 1}, AJUSTE O FOCO, e pressione ENTER...")

        x_atual = core.get_x_position(xy_stage)
        y_atual = core.get_y_position(xy_stage)
        z_atual = core.get_position(z_stage)

        pontos_coletados.append((x_atual, y_atual, z_atual))

        print(f"-> Ponto {p + 1} salvo: X={x_atual:.1f}, Y={y_atual:.1f}, Z={z_atual:.2f}")

    return pontos_coletados


def gerar_malha_varredura(core, xy_stage, fov_x, fov_y):
    """Pede o ponto inicial e final da área a varrer e gera a malha de
    posições (X, Y) cobrindo essa área, espaçadas pelo campo de visão
    (fov_x, fov_y) da objetiva. Retorna um dicionário com a malha e os
    dados usados para calculá-la."""

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

    qtd_passos_x = math.ceil(dist_x / fov_x)
    qtd_passos_y = math.ceil(dist_y / fov_y)

    malha_x = [x_inicial + (i * fov_x * direcao_x) for i in range(qtd_passos_x)]
    malha_y = [y_inicial + (j * fov_y * direcao_y) for j in range(qtd_passos_y)]

    return {
        "malha_x": malha_x,
        "malha_y": malha_y,
        "x_inicial": x_inicial,
        "y_inicial": y_inicial,
        "dist_x": dist_x,
        "dist_y": dist_y,
        "qtd_passos_x": qtd_passos_x,
        "qtd_passos_y": qtd_passos_y,
    }
