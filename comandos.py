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
# Isso não é hipotético. Seis pontos onde quatro são os cantos de um
# retângulo dão cond ~1e7 no ajuste quadrático (nos quatro cantos, dx² e
# dy² valem o mesmo, então essas colunas ficam quase constantes e quase
# dependentes da coluna de 1s). Um teste de posto NÃO pega isso: o posto
# dá 6, cheio, e o ajuste passa devolvendo lixo. Com 8+ pontos bem
# espalhados o cond fica na casa das dezenas.
LIMITE_CONDICIONAMENTO = 1e4


def calibrar_plano_foco(pontos, verboso=True):
    """Ajusta um plano de foco da forma:
        Z = Z0 + ax*(X - X0) + ay*(Y - Y0).

    verboso=False silencia o resumo impresso -- usado no re-ajuste
    automático durante a varredura, que roda dezenas de vezes e encheria
    o console de tabelas."""

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

    if len(pontos) < 3:
        raise ValueError(
            "ERRO: São necessários pelo menos 3 pontos para ajustar o plano de foco."
        )

    matriz = np.column_stack([dx, dy])

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

    xs = pontos[:, 0]
    ys = pontos[:, 1]
    zs = pontos[:, 2]

    x0 = np.mean(xs)
    y0 = np.mean(ys)

    dx = xs - x0
    dy = ys - y0

    # Normaliza as coordenadas ANTES de montar a matriz de projeto, para
    # que as colunas fiquem na mesma ordem de grandeza: sem isso, numa área
    # de ~500 µm, dx fica nas centenas e dx² nas centenas de milhares.
    # Isso melhora o condicionamento cerca de 25x (medido: 2,5e8 -> 9,8e6).
    # Sozinho NÃO resolve: o que domina o condicionamento é a geometria dos
    # pontos, não a escala das unidades -- ver LIMITE_CONDICIONAMENTO e a
    # verificação logo abaixo.
    escala = float(max(np.abs(dx).max(), np.abs(dy).max()))

    if escala == 0:
        raise ValueError(
            "ERRO: todos os pontos de calibração estão na mesma posição XY."
        )

    ux = dx / escala
    uy = dy / escala

    if len(pontos) < 6:
        raise ValueError(
            "ERRO: A superfície quadrática tem 6 coeficientes e precisa de pelo "
            "menos 6 pontos. Na prática, use mais -- com exatamente 6 o ajuste "
            "passa exatamente por eles e reproduz o erro de cada um sem nenhuma "
            "média para amortecê-lo."
        )

    matriz_teste = np.column_stack([ux, uy, ux * uy, ux**2, uy**2, np.ones(len(pontos))])

    condicionamento = np.linalg.cond(matriz_teste)

    if not np.isfinite(condicionamento) or condicionamento > LIMITE_CONDICIONAMENTO:
        raise ValueError(
            "ERRO: Os pontos de calibração estão mal distribuídos para a "
            f"superfície quadrática (condicionamento {condicionamento:.1e}, "
            f"limite {LIMITE_CONDICIONAMENTO:.0e}). Nesse estado, o erro ao "
            "focar cada ponto seria amplificado por esse fator no Z previsto. "
            "Use mais pontos e evite colocá-los só nos cantos e no centro -- "
            "espalhe-os de forma irregular pela área da varredura."
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
