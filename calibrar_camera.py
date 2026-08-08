"""
Calibra a camera do microscópio (balanço de branco + exposição)

Balanço de branco: aponte para uma área neutra da amostra (substrato
limpo, sem flake) e o script ajusta WhiteBalanceRed/WhiteBalanceBlue até
essa área aparecer realmente cinza (R, G e B com médias parecidas).

Exposição: testa uma lista de exposições candidatas e escolhe a que
melhor equilibra "não ficar escura demais" com "não estourar o brilho",
olhando direto o histograma de intensidade da imagem.

No final, o script mostra os valores encontrados -- copie-os para o parâmetros.py
"""

from pycromanager import Core

import numpy as np

from capturar_imagem import capturar_imagem

# Exposições a testar, em ms. Ajuste essa lista conforme a faixa que faz sentido para objetiva.
EXPOSICOES = list(range(3,26,1))

# Exposição usada só durante a etapa de balanço de branco
EXPOSICAO_PARA_BALANCO = 10.0

# Quanto da correção calculada a cada rodada é realmente aplicada.
TAXA_APRENDIZADO = 0.5

# Quando a diferença entre os canais for menor que isso (em fração, ex: 0.01 = 1%), consideramos que o balanço de branco convergiu.
TOLERANCIA_BALANCO = 0.01

MAX_ITERACOES_BALANCO = 100


def limitar_ao_intervalo_permitido(core, camera, nome_propriedade, valor):
    """Garante que o valor não passe dos limites que a câmera aceita."""

    if core.has_property_limits(camera, nome_propriedade):
        minimo = core.get_property_lower_limit(camera, nome_propriedade)
        maximo = core.get_property_upper_limit(camera, nome_propriedade)
        valor = max(minimo, min(maximo, valor))

    return valor


def _sair_do_zero_se_preciso(core, camera, nome_propriedade, valor_atual):
    """O ajuste em calibrar_balanco_de_branco é multiplicativo (valor * (1 +
    taxa*erro)): se valor_atual for exatamente 0, a correção fica travada em
    0 pra sempre, não importa o erro (0 vezes qualquer coisa é 0)."""

    if valor_atual != 0:
        return valor_atual

    if core.has_property_limits(camera, nome_propriedade):
        return core.get_property_upper_limit(camera, nome_propriedade) * 0.01

    return 1.0


def calibrar_balanco_de_branco(core, camera):
    """
    Ajusta WhiteBalanceRed/WhiteBalanceBlue por tentativa proporcional:
    mede o quanto R e B estão desequilibrados em relação a G, corrige na
    direção certa, mede de novo, repete até convergir.

    Isso evita ter que saber de antemão a escala exata da propriedade da
    câmera -- só assume que "aumentar o valor da propriedade aumenta a
    presença daquele canal na imagem", que é o comportamento usual desse
    tipo de propriedade.
    """

    for iteracao in range(1, MAX_ITERACOES_BALANCO + 1):
        imagem = capturar_imagem(core, camera)

        if imagem.ndim != 3:
            print(
                "[Erro] A câmera não está devolvendo uma imagem colorida "
                "(RGB) agora -- balanço de branco não se aplica em modo "
                "monocromático. Abortando essa etapa."
            )
            return None, None

        media_r = float(imagem[:, :, 0].mean())
        media_g = float(imagem[:, :, 1].mean())
        media_b = float(imagem[:, :, 2].mean())

        erro_r = (media_g - media_r) / media_g
        erro_b = (media_g - media_b) / media_g

        print(
            f"[{iteracao:02d}] R={media_r:6.1f}  G={media_g:6.1f}  B={media_b:6.1f}  "
            f"(erro R={erro_r:+.1%}, erro B={erro_b:+.1%})"
        )

        if abs(erro_r) < TOLERANCIA_BALANCO and abs(erro_b) < TOLERANCIA_BALANCO:
            print("[Sucesso] Balanço de branco convergiu.\n")
            break

        valor_r_atual = float(core.get_property(camera, "WhiteBalanceRed"))
        valor_b_atual = float(core.get_property(camera, "WhiteBalanceBlue"))

        valor_r_atual = _sair_do_zero_se_preciso(core, camera, "WhiteBalanceRed", valor_r_atual)
        valor_b_atual = _sair_do_zero_se_preciso(core, camera, "WhiteBalanceBlue", valor_b_atual)

        novo_r = valor_r_atual * (1 + TAXA_APRENDIZADO * erro_r)
        novo_b = valor_b_atual * (1 + TAXA_APRENDIZADO * erro_b)

        novo_r = limitar_ao_intervalo_permitido(core, camera, "WhiteBalanceRed", novo_r)
        novo_b = limitar_ao_intervalo_permitido(core, camera, "WhiteBalanceBlue", novo_b)

        core.set_property(camera, "WhiteBalanceRed", novo_r)
        core.set_property(camera, "WhiteBalanceBlue", novo_b)
        core.wait_for_device(camera)
    else:
        print(
            "[Aviso] Não convergiu dentro do limite de iterações. "
            "Os últimos valores testados foram mantidos -- confira se "
            "fazem sentido antes de usar.\n"
        )

    valor_r_final = core.get_property(camera, "WhiteBalanceRed")
    valor_b_final = core.get_property(camera, "WhiteBalanceBlue")

    return valor_r_final, valor_b_final


def escolher_exposicao(core, camera, candidatas):
    """
    Testa cada exposição candidata e escolhe a que melhor equilibra
    "poucos pixels escuros demais" com "poucos pixels estourados",
    usando o histograma de intensidade da imagem
    """

    resultados = []

    for exp in candidatas:
        core.set_exposure(exp)
        core.wait_for_device(camera)

        imagem = capturar_imagem(core, camera)

        fracao_escura = float(np.mean(imagem <= 5))
        fracao_estourada = float(np.mean(imagem >= 250))
        pontuacao = fracao_escura + fracao_estourada  # quanto menor, melhor

        resultados.append((exp, fracao_escura, fracao_estourada, pontuacao))

        print(
            f"Exposição {exp:>5.1f} ms | "
            f"escura: {fracao_escura:6.1%} | "
            f"estourada: {fracao_estourada:6.1%} | "
            f"pontuação: {pontuacao:.4f}"
        )

    melhor = min(resultados, key=lambda r: r[3])
    exposicao_escolhida = melhor[0]

    core.set_exposure(exposicao_escolhida)
    core.wait_for_device(camera)

    print(f"\n[Sucesso] Exposição escolhida: {exposicao_escolhida} ms\n")

    return exposicao_escolhida


def main():
    core = Core()
    camera = core.get_camera_device()

    if core.is_sequence_running():
        core.stop_sequence_acquisition()

    input(
        "Mova o estágio para uma área NEUTRA da amostra "
        "(substrato limpo, sem flake) e pressione ENTER..."
    )

    core.set_exposure(EXPOSICAO_PARA_BALANCO)
    core.wait_for_device(camera)

    print("\n--- CALIBRANDO BALANÇO DE BRANCO ---")
    valor_r, valor_b = calibrar_balanco_de_branco(core, camera)

    print("--- ESCOLHENDO EXPOSIÇÃO ---")
    exposicao_escolhida = escolher_exposicao(core, camera, EXPOSICOES)

    print("--- RESULTADO FINAL ---")
    print(f"WhiteBalanceRed  = {valor_r}")
    print(f"WhiteBalanceBlue = {valor_b}")
    print(f"Exposição        = {exposicao_escolhida} ms")
    print(
        "\nCopie esses valores para o parametros.py"
    )


if __name__ == "__main__":
    main()
