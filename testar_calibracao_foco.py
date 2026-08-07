"""Testes de regressão da calibração de foco -- rodam SEM microscópio.

Cobrem duas coisas do comandos.py: a escolha da geometria dos pontos de
calibração (sugerir_pontos_calibracao e companhia) e o fluxo guiado de
coleta, com um estágio falso no lugar do Core do Micro-Manager.

Por que dá para testar isso numa máquina qualquer: o condicionamento da
matriz de projeto depende só de X e Y, nunca de Z. Toda a parte que decide
SE um conjunto de pontos serve é geometria pura. O que sobra -- se o
estágio de fato vai para onde foi mandado -- é o que só o microscópio
responde, e não é o que este arquivo tenta cobrir.

Rode com:
    python testar_calibracao_foco.py

Sai com código 1 se algo falhar, para poder ser encadeado.
"""

import builtins
import io
import math
import sys
from contextlib import redirect_stdout

import numpy as np

from comandos import (
    LIMITE_CONDICIONAMENTO,
    _condicionamento,
    _melhor_grid,
    calcular_z,
    calibrar_plano_foco,
    calibrar_superficie_foco,
    coletar_pontos_calibracao,
    sugerir_pontos_calibracao,
)

falhas = []


def checar(condicao, mensagem):
    """Registra uma falha em vez de levantar, para que uma quebra não
    esconda o resultado dos testes seguintes."""

    if not condicao:
        falhas.append(mensagem)
        print(f"    FALHOU: {mensagem}")


def malha_falsa(x0, y0, passo_x, passo_y, qtd_x, qtd_y):
    """Monta o mínimo do dicionário que gerar_malha_varredura devolve --
    só as chaves que sugerir_pontos_calibracao lê."""

    return {
        "malha_x": [x0 + i * passo_x for i in range(qtd_x)],
        "malha_y": [y0 + j * passo_y for j in range(qtd_y)],
        "passo_x": passo_x,
        "passo_y": passo_y,
    }


# Malha de referência usada na maioria dos testes: ~1000 x 1000 µm.
MALHA = malha_falsa(1000.0, 2000.0, 90.0, 90.0, 12, 12)

# Superfícies de amostra simuladas. "Focar" um ponto é ler o Z da
# superfície ali -- é o usuário perfeito, sem erro de foco.
PLANO_REAL = lambda x, y: 50.0 + 0.002 * x - 0.0015 * y
QUADRATICA_REAL = lambda x, y: PLANO_REAL(x, y) + 1e-6 * (x - 1500.0) ** 2


class EstagioFalso:
    """Substitui o Core do Micro-Manager.

    set_position levanta de propósito: o script NÃO pode mexer no Z
    durante a coleta (ver coletar_pontos_calibracao), e um teste que só
    contasse as chamadas deixaria passar uma regressão silenciosa."""

    def __init__(self, superficie, desvio=0.0):
        self.x = 0.0
        self.y = 0.0
        self.superficie = superficie
        # Quanto o "usuário" anda depois que o estágio chega ao alvo --
        # simula procurar uma região com estrutura em que dê para focar.
        self.desvio = desvio

    def set_xy_position(self, dispositivo, x, y):
        self.x = x + self.desvio
        self.y = y + self.desvio

    def wait_for_device(self, dispositivo):
        pass

    def get_x_position(self, dispositivo):
        return self.x

    def get_y_position(self, dispositivo):
        return self.y

    def get_position(self, dispositivo):
        return self.superficie(self.x, self.y)

    def set_position(self, dispositivo, z):
        raise AssertionError("o script mexeu no Z durante a coleta")


def rodar_coleta(qtd, tipo, malha, superficie, desvio=0.0):
    """Roda coletar_pontos_calibracao com o estágio falso, respondendo aos
    input() no lugar do usuário. Devolve (pontos, saída impressa)."""

    estagio = EstagioFalso(superficie, desvio)

    # Primeira resposta: a quantidade de pontos. Depois, um ENTER por
    # ponto ("já ajustei o foco").
    respostas = iter([str(qtd)] + [""] * qtd)
    input_original = builtins.input
    builtins.input = lambda *args, **kwargs: next(respostas)

    saida = io.StringIO()

    try:
        with redirect_stdout(saida):
            pontos = coletar_pontos_calibracao(
                estagio, "xy", "z", 3, malha=malha, tipo=tipo
            )
    finally:
        builtins.input = input_original

    return pontos, saida.getvalue()


def testar_melhor_grid():
    """A escolha do grid: maior que cabe, mais quadrado possível, com o
    mínimo de níveis em cada eixo."""

    print("\n1. _melhor_grid")

    quadratica = [_melhor_grid(qtd, 3) for qtd in (10, 12, 16)]
    plano = [_melhor_grid(qtd, 2) for qtd in (3, 4, 9)]

    print(f"    quadrática (10, 12, 16): {quadratica}")
    print(f"    plano (3, 4, 9):         {plano}")

    checar(quadratica == [(3, 3), (3, 4), (4, 4)], f"grid da quadrática: {quadratica}")

    # 3 pontos no plano não cabem em nenhum grid (o menor, 2x2, já são 4)
    # -- é o caso que cai no triângulo largo.
    checar(plano == [None, (2, 2), (3, 3)], f"grid do plano: {plano}")


def testar_condicionamento_das_sugestoes():
    """Nenhuma quantidade de pontos pode gerar um arranjo que o próprio
    ajuste depois recusaria."""

    print("\n2. condicionamento dos arranjos sugeridos")

    for tipo, minimo in (("plano", 3), ("quadratica", 10)):
        condicionamentos = []

        for qtd in range(minimo, 21):
            sugestao = sugerir_pontos_calibracao(MALHA, qtd, tipo)

            checar(
                len(sugestao["pontos"]) == qtd,
                f"{tipo} com {qtd}: devolveu {len(sugestao['pontos'])} pontos",
            )
            checar(
                len(set(sugestao["pontos"])) == qtd,
                f"{tipo} com {qtd}: há pontos repetidos",
            )
            checar(
                sugestao["condicionamento"] < LIMITE_CONDICIONAMENTO,
                f"{tipo} com {qtd}: cond {sugestao['condicionamento']:.1e} estourou o limite",
            )

            condicionamentos.append(sugestao["condicionamento"])

        print(
            f"    {tipo:11s}: cond de {min(condicionamentos):.2f} a "
            f"{max(condicionamentos):.2f} entre {minimo} e 20 pontos"
        )


def testar_niveis_por_eixo():
    """A regra que evita a singularidade: grau d exige d+1 valores
    DISTINTOS da coordenada naquele eixo."""

    print("\n3. níveis distintos por eixo")

    for tipo, minimo_niveis, quantidades in (
        ("plano", 2, (4, 10, 16)),
        ("quadratica", 3, (10, 12, 16)),
    ):
        for qtd in quantidades:
            pontos = sugerir_pontos_calibracao(MALHA, qtd, tipo)["pontos"]

            niveis_x = len({round(ponto[0], 6) for ponto in pontos})
            niveis_y = len({round(ponto[1], 6) for ponto in pontos})

            suficiente = niveis_x >= minimo_niveis and niveis_y >= minimo_niveis

            print(
                f"    {tipo:11s} {qtd:2d} pontos: {niveis_x} níveis em X, "
                f"{niveis_y} em Y ({'ok' if suficiente else 'INSUFICIENTE'})"
            )
            checar(suficiente, f"{tipo} com {qtd}: níveis {niveis_x}x{niveis_y}")


def testar_coluna_unica():
    """Varredura de uma coluna só de tiles: sem tratamento, todos os alvos
    cairiam no mesmo X e a matriz seria singular."""

    print("\n4. varredura de uma coluna só")

    malha = malha_falsa(500.0, 100.0, 90.0, 90.0, 1, 15)

    for tipo, qtd in (("plano", 4), ("quadratica", 12)):
        sugestao = sugerir_pontos_calibracao(malha, qtd, tipo)
        niveis_x = len({round(ponto[0], 6) for ponto in sugestao["pontos"]})

        print(f"    {tipo:11s}: {niveis_x} níveis em X, cond {sugestao['condicionamento']:.2f}")

        checar(
            np.isfinite(sugestao["condicionamento"])
            and sugestao["condicionamento"] < LIMITE_CONDICIONAMENTO,
            f"coluna única, {tipo}: cond {sugestao['condicionamento']:.1e}",
        )


def testar_area_alongada():
    """Numa área 10:1 o condicionamento sobe por causa da FORMA da área, e
    não da posição dos pontos. Tem que continuar passando, e o aviso da
    coleta tem que disparar."""

    print("\n5. área alongada (10:1)")

    malha = malha_falsa(0.0, 0.0, 100.0, 100.0, 51, 6)

    for tipo, qtd in (("plano", 10), ("quadratica", 12)):
        sugestao = sugerir_pontos_calibracao(malha, qtd, tipo)
        avisaria = sugestao["condicionamento"] > LIMITE_CONDICIONAMENTO / 100

        print(
            f"    {tipo:11s}: cond {sugestao['condicionamento']:.1f}"
            f"{'  (dispara aviso)' if avisaria else ''}"
        )
        checar(
            sugestao["condicionamento"] < LIMITE_CONDICIONAMENTO,
            f"área alongada, {tipo}: cond {sugestao['condicionamento']:.1e}",
        )


def testar_serpentina():
    """A ordem de visita tem que encurtar o caminho do estágio."""

    print("\n6. ordem de visita")

    pontos = sugerir_pontos_calibracao(MALHA, 16, "quadratica")["pontos"]

    caminho = sum(math.dist(pontos[k], pontos[k + 1]) for k in range(len(pontos) - 1))

    lexicografica = sorted(pontos)
    caminho_ingenuo = sum(
        math.dist(lexicografica[k], lexicografica[k + 1])
        for k in range(len(lexicografica) - 1)
    )

    print(f"    serpentina: {caminho:.0f} µm | ordem lexicográfica: {caminho_ingenuo:.0f} µm")
    checar(caminho <= caminho_ingenuo, "a serpentina não encurtou o caminho")


def testar_fluxo_guiado():
    """Coleta guiada de ponta a ponta: o estágio anda, o usuário foca, o
    modelo ajustado tem que reproduzir a superfície simulada."""

    print("\n7. fluxo guiado de ponta a ponta")

    casos = (
        ("plano, 3 pontos (triângulo)", "plano", 3, PLANO_REAL, calibrar_plano_foco),
        ("plano, 10 pontos", "plano", 10, PLANO_REAL, calibrar_plano_foco),
        ("quadrática, 12 pontos", "quadratica", 12, QUADRATICA_REAL, calibrar_superficie_foco),
    )

    for rotulo, tipo, qtd, superficie, calibrador in casos:
        pontos, _ = rodar_coleta(qtd, tipo, MALHA, superficie)

        with redirect_stdout(io.StringIO()):
            modelo = calibrador(pontos, verboso=False)

        # Como o "usuário" foca sem erro nenhum, o ajuste tem que
        # reproduzir a superfície até a precisão do float. Qualquer erro
        # visível aqui é bug de código, não imprecisão de medida.
        erro_maximo = max(
            abs(superficie(x, y) - calcular_z(x, y, modelo)) for x, y, _ in pontos
        )

        print(f"    {rotulo:28s} {len(pontos):2d} pontos | erro máx {erro_maximo:.2e} µm")

        checar(len(pontos) == qtd, f"{rotulo}: coletou {len(pontos)} pontos")
        checar(erro_maximo < 1e-6, f"{rotulo}: erro de {erro_maximo:.2e} µm no ajuste")


def testar_aviso_de_desvio():
    """O aviso só deve aparecer quando o usuário se afasta MAIS que o
    espaçamento do arranjo. Nesta malha o grid 3x4 tem espaçamento de
    ~330 µm, então 200 µm passa calado e 400 µm avisa."""

    print("\n8. aviso de desvio do alvo")

    for desvio, deve_avisar in ((200.0, False), (400.0, True)):
        pontos, saida = rodar_coleta(12, "quadratica", MALHA, QUADRATICA_REAL, desvio=desvio)

        condicionamento = _condicionamento(
            [ponto[0] for ponto in pontos], [ponto[1] for ponto in pontos], "quadratica"
        )
        avisou = "[Aviso] desvio" in saida

        print(
            f"    desvio de {desvio:.0f} µm: cond {condicionamento:.2f}, "
            f"avisou={avisou} (esperado {deve_avisar})"
        )

        checar(
            condicionamento < LIMITE_CONDICIONAMENTO,
            f"desvio de {desvio:.0f} µm: cond {condicionamento:.1e}",
        )
        checar(avisou == deve_avisar, f"aviso errado com desvio de {desvio:.0f} µm")


def testar_modo_manual():
    """Sem malha, a coleta tem que voltar ao comportamento antigo: o
    usuário move o estágio, o script não move nada."""

    print("\n9. modo manual (sem malha)")

    class EstagioImovel(EstagioFalso):
        def set_xy_position(self, dispositivo, x, y):
            raise AssertionError("o script moveu XY no modo manual")

    estagio = EstagioImovel(PLANO_REAL)

    respostas = iter(["4", "", "", "", ""])
    input_original = builtins.input
    builtins.input = lambda *args, **kwargs: next(respostas)

    try:
        with redirect_stdout(io.StringIO()):
            pontos = coletar_pontos_calibracao(estagio, "xy", "z", 3)
    finally:
        builtins.input = input_original

    print(f"    coletou {len(pontos)} pontos sem mover o estágio")
    checar(len(pontos) == 4, f"modo manual: coletou {len(pontos)} pontos")


def main():
    print("=" * 68)
    print("TESTES DE CALIBRAÇÃO DE FOCO (sem microscópio)")
    print("=" * 68)

    testar_melhor_grid()
    testar_condicionamento_das_sugestoes()
    testar_niveis_por_eixo()
    testar_coluna_unica()
    testar_area_alongada()
    testar_serpentina()
    testar_fluxo_guiado()
    testar_aviso_de_desvio()
    testar_modo_manual()

    print()
    print("=" * 68)

    if falhas:
        print(f"RESULTADO: {len(falhas)} falha(s)")
        for falha in falhas:
            print(f"  - {falha}")
    else:
        print("RESULTADO: tudo passou")

    print("=" * 68)

    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
