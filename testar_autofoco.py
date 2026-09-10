"""Testes da busca de foco e da robustez do modelo -- rodam SEM microscópio.

O que este arquivo cobre, e por que dá para cobrir numa máquina qualquer:

    buscar_foco          precisa de um estágio Z e de uma câmera. Os dois
                         são simulados aqui por um MicroscopioFalso que
                         devolve imagens REALMENTE desfocadas, borradas na
                         medida em que o Z está longe do foco daquela
                         posição. Quem mede a nitidez continua sendo a
                         medir_nitidez de verdade -- o que se simula é a
                         ótica, não a métrica.
    ordem_serpentina     combinatória pura.
    pesos_por_idade      aritmética pura.
    checkpoint_autofoco  o caminho completo, incluindo a rejeição de
                         medidas espúrias, com um detrito plantado de
                         propósito no meio da amostra.

O que ele NÃO cobre, e nenhum teste offline cobriria: se o estágio de
verdade assenta antes da foto, se a curva de nitidez da SUA amostra tem
mesmo um pico só, e quanto tempo cada etapa custa no seu equipamento.
Isso é o que o cronômetro da varredura mede.

Rode com:
    python testar_autofoco.py

Sai com código 1 se algo falhar, para poder ser encadeado.
"""

import math
import sys

import cv2
import numpy as np

from comandos import (
    DESLOCAMENTO_MINIMO_REFINO,
    MEIA_VIDA_PONTOS,
    PESO_MINIMO_ANCORA,
    buscar_foco,
    calcular_z,
    calibrar_plano_foco,
    checkpoint_autofoco,
    desvio_robusto,
    medir_nitidez,
    medir_textura,
    ordem_serpentina,
    pesos_por_idade,
    reajustar_modelo_foco,
)

falhas = []


def checar(condicao, mensagem):
    """Registra a falha em vez de levantar, para que uma quebra não esconda
    o resultado dos testes seguintes."""

    if not condicao:
        falhas.append(mensagem)
        print(f"    FALHOU: {mensagem}")


# =====================================================================
#  O MICROSCÓPIO FALSO
# =====================================================================

# Lado da imagem simulada. Pequena de propósito: a suíte tira centenas de
# fotos e o que se testa aqui não depende da resolução.
LADO = 256

# Desfoque residual no foco perfeito, em pixels -- o análogo do limite de
# difração. Sem ele o pico seria uma quina, e a interpolação parabólica
# estaria sendo testada contra uma curva que a ótica real nunca produz.
SIGMA_NO_FOCO = 1.5

# Quantos µm de erro de Z valem um pixel de desfoque. Estes dois números
# foram calibrados juntos para que a curva de nitidez saia com largura a
# 90% da altura de ~4,5 µm: a profundidade de campo implícita na faixa de
# 10 µm do 50x, já que o parametros.py recomenda faixa ~ 2,5 x DOF.
#
# Isso importa mais do que parece. Com a curva estreita demais (o primeiro
# chute deste arquivo dava 1 µm) o passo de 1 µm subamostra o pico, os dois
# vizinhos do máximo saem empatados e o teste de proeminência recusa TODA
# medida -- inclusive as boas. É o comportamento correto do código diante
# de uma busca subamostrada, mas seria o simulador sendo testado, não o
# autofoco. O teste 1 existe para prender essa calibração no lugar.
UM_POR_PIXEL_DE_DESFOQUE = 6.0


def textura_base(semente=0):
    """Um campo com estrutura em várias escalas: ruído branco suavizado.

    Suavizado, e não ruído puro, porque ruído branco tem toda a energia na
    escala de 1 px -- ele desapareceria ao primeiro blur e a curva de
    nitidez cairia a pique, o que não se parece com uma amostra."""

    gerador = np.random.default_rng(semente)
    campo = gerador.normal(0.0, 1.0, (LADO, LADO)).astype(np.float32)
    campo = cv2.GaussianBlur(campo, (0, 0), 1.6)

    # Normaliza para ocupar bem os 8 bits, senão a quantização domina.
    campo = campo - campo.min()
    campo = campo / campo.max() * 255.0

    return campo.astype(np.uint8)


class MicroscopioFalso:
    """O mínimo do Core do Micro-Manager que o autofoco usa.

    `foco_real` é uma função (x, y) -> Z em foco: é ela que define a
    "amostra". Um plano inclinado, uma superfície curva ou um plano com um
    detrito num canto são só três funções diferentes.
    """

    def __init__(self, foco_real, semente=0, contraste=1.0):
        self.foco_real = foco_real
        self.base = textura_base(semente)
        self.contraste = contraste

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        # Quantas fotos foram tiradas desde o último zerar_contagem: é o
        # número que a busca expansiva existe para reduzir.
        self.fotos = 0

    # --- estágio -----------------------------------------------------
    def set_position(self, dispositivo, z):
        self.z = float(z)

    def get_position(self, dispositivo):
        return self.z

    def set_xy_position(self, dispositivo, x, y):
        self.x = float(x)
        self.y = float(y)

    def get_x_position(self, dispositivo):
        return self.x

    def get_y_position(self, dispositivo):
        return self.y

    def wait_for_device(self, dispositivo):
        pass

    # --- câmera ------------------------------------------------------
    def snap_image(self):
        self.fotos += 1

        erro = self.z - self.foco_real(self.x, self.y)

        # Desfoque em quadratura: no foco sobra SIGMA_NO_FOCO, e longe dele
        # o termo linear domina. É o que dá ao pico uma curvatura finita em
        # vez de um bico.
        sigma = math.hypot(SIGMA_NO_FOCO, erro / UM_POR_PIXEL_DE_DESFOQUE)

        imagem = cv2.GaussianBlur(self.base, (0, 0), sigma)

        if self.contraste != 1.0:
            imagem = np.clip(
                128 + (imagem.astype(np.float32) - 128) * self.contraste, 0, 255
            ).astype(np.uint8)

        # A câmera real entrega BGRA achatado; a capturar_imagem remonta.
        bgra = np.zeros((LADO, LADO, 4), dtype=np.uint8)
        bgra[:, :, 0] = imagem
        bgra[:, :, 1] = imagem
        bgra[:, :, 2] = imagem

        self._ultimo = bgra.reshape(-1)

    def get_image(self):
        return self._ultimo

    def get_image_width(self):
        return LADO

    def get_image_height(self):
        return LADO

    def get_bytes_per_pixel(self):
        return 1

    def zerar_contagem(self):
        self.fotos = 0


def plano(z0=100.0, ax=0.0, ay=0.0):
    """Uma amostra plana, opcionalmente inclinada."""

    return lambda x, y: z0 + ax * x + ay * y


# =====================================================================
#  1. A BUSCA DE FOCO
# =====================================================================

FAIXA = 10.0
PASSO = 1.0

# Custo da varredura completa da faixa, que é o que a busca antiga gastava
# em TODO checkpoint. Serve de referência para o ganho.
FOTOS_DA_VARREDURA_COMPLETA = int(2 * FAIXA / PASSO + 1)


def testar_forma_da_curva():
    """Antes de testar a busca, confere que a ótica simulada produz uma
    curva parecida com a de um microscópio: um pico só, e com largura
    compatível com a faixa configurada. Se este teste falhar, todos os
    outros deste bloco estão medindo o simulador, não o código."""

    print("\n1. forma da curva de nitidez simulada")

    microscopio = MicroscopioFalso(plano(z0=100.0))

    zs = np.arange(100.0 - FAIXA, 100.0 + FAIXA + 0.25, 0.25)
    nitidezes = []

    for z in zs:
        microscopio.set_position("z", z)
        microscopio.snap_image()

        bgra = microscopio.get_image().reshape((LADO, LADO, 4))
        nitidezes.append(medir_nitidez(bgra[:, :, 0]))

    nitidezes = np.array(nitidezes)
    pico = nitidezes.max()

    # Largura a 90% da altura: a definição de DOF efetiva que o
    # parametros.py usa para deduzir faixa e passo.
    acima = zs[nitidezes >= 0.9 * pico]
    largura = acima.max() - acima.min()

    # Unimodal: a curva sobe até o pico e desce, sem voltar a subir.
    #
    # A monotonia é testada em sentido FRACO (>= e <=) porque o simulador
    # trabalha em uint8: perto do topo, dois Z separados por menos de meio
    # micrômetro produzem a mesma imagem depois de arredondada, e a curva
    # anda em degraus. Um platô é a quantização; o que não pode aparecer é
    # a curva voltando a subir depois do pico.
    indice_pico = int(np.argmax(nitidezes))
    subida = np.all(np.diff(nitidezes[: indice_pico + 1]) >= 0)
    descida = np.all(np.diff(nitidezes[indice_pico:]) <= 0)

    print(f"    pico em Z={zs[indice_pico]:.2f} | largura a 90%: {largura:.2f} µm")
    print(f"    unimodal: sobe={subida}, desce={descida}")

    checar(abs(zs[indice_pico] - 100.0) < 0.3, "o pico simulado não está no foco")
    checar(subida and descida, "a curva simulada não é unimodal")
    # A faixa configurada supõe DOF ~ faixa/2,5 = 4 µm. Uma curva muito
    # mais estreita que isso subamostraria o pico (ver
    # UM_POR_PIXEL_DE_DESFOQUE); muito mais larga faria a busca parecer
    # mais fácil do que é.
    checar(3.0 < largura < 6.0, f"largura a 90% de {largura:.2f} µm fora do plausível")


def testar_busca_acha_o_pico():
    """O caso normal: o modelo erra por menos que a faixa, e a busca tem
    que achar o foco com precisão melhor que o passo -- gastando bem menos
    que a varredura completa."""

    print("\n2. a busca encontra o foco (e o que ela custa)")

    z_real = 100.0
    microscopio = MicroscopioFalso(plano(z0=z_real))

    print(f"    {'erro do modelo':>16}{'erro final':>13}{'fotos':>8}{'queda':>9}")

    for erro_do_modelo in (0.0, 0.4, 1.7, -3.2, 6.5, -8.0):
        microscopio.zerar_contagem()

        resultado = buscar_foco(
            microscopio, "z", "camera", z_real + erro_do_modelo, FAIXA, PASSO
        )

        erro_final = resultado["z"] - z_real

        print(
            f"    {erro_do_modelo:>+13.1f} µm{erro_final:>+10.2f} µm"
            f"{resultado['fotos']:>8}{resultado['queda']:>8.1%}"
        )

        checar(
            resultado["confiavel"],
            f"erro de {erro_do_modelo:+.1f} µm: medida recusada dentro da faixa",
        )
        # Meio passo é o que o argmax puro garantiria; a interpolação tem
        # de fazer melhor que isso.
        checar(
            abs(erro_final) < PASSO / 2,
            f"erro de {erro_do_modelo:+.1f} µm: sobrou {erro_final:+.2f} µm de erro",
        )
        checar(
            resultado["fotos"] < FOTOS_DA_VARREDURA_COMPLETA,
            f"erro de {erro_do_modelo:+.1f} µm: gastou {resultado['fotos']} fotos, "
            f"o mesmo que a varredura completa",
        )


def testar_custo_quando_o_modelo_acerta():
    """O caso que domina uma varredura bem calibrada: o modelo já está
    certo. É aqui que estava o desperdício -- 21 fotos para confirmar que
    não havia nada a corrigir."""

    print("\n3. custo com o modelo certo")

    microscopio = MicroscopioFalso(plano(z0=100.0))
    microscopio.zerar_contagem()

    resultado = buscar_foco(microscopio, "z", "camera", 100.0, FAIXA, PASSO)

    print(
        f"    {resultado['fotos']} fotos "
        f"(varredura completa da faixa: {FOTOS_DA_VARREDURA_COMPLETA})"
    )

    # 3 medidas + no máximo 1 foto de refino no vértice.
    checar(resultado["fotos"] <= 4, f"gastou {resultado['fotos']} fotos com o modelo certo")
    checar(resultado["confiavel"], "recusou a medida com o modelo certo")


def testar_refino_parabolico():
    """A interpolação do vértice tem que ganhar do argmax puro. Comparação
    direta: o Z devolvido contra o Z da amostra mais nítida da grade.

    Rodado com a faixa e o passo do 20x (30 µm / 3 µm), e não com os do
    50x usados no resto do arquivo, por uma limitação do SIMULADOR: as
    imagens são uint8, e uma diferença de desfoque abaixo de ~0,5 µm não
    move nenhum pixel depois de arredondado. Com passo de 1 µm os
    deslocamentos sub-passo a testar cairiam todos dentro dessa cegueira;
    com passo de 3 µm eles são resolvidos. A ótica de verdade não tem esse
    degrau -- ela é contínua."""

    print("\n4. o refino sub-passo ganha do argmax")

    FAIXA_20X = 30.0
    PASSO_20X = 3.0

    erros_com_refino = []
    erros_sem_refino = []

    # Deslocamentos sub-passo variados: é onde o argmax puro sofre, porque
    # o pico verdadeiro cai entre duas amostras da grade.
    for fracao in (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9):
        z_real = 100.0
        z_estimado = z_real - fracao * PASSO_20X

        microscopio = MicroscopioFalso(plano(z0=z_real))
        resultado = buscar_foco(
            microscopio, "z", "camera", z_estimado, FAIXA_20X, PASSO_20X
        )

        checar(
            resultado["confiavel"],
            f"fração {fracao}: medida recusada com o foco dentro da faixa",
        )

        # O Z sem refino é o da amostra escolhida, ou seja, o resultado já
        # descontado o deslocamento sub-passo.
        z_sem_refino = resultado["z"] - resultado["delta_passos"] * PASSO_20X

        erros_com_refino.append(abs(resultado["z"] - z_real))
        erros_sem_refino.append(abs(z_sem_refino - z_real))

    com = float(np.mean(erros_com_refino))
    sem = float(np.mean(erros_sem_refino))

    print(
        f"    passo de {PASSO_20X} µm | erro médio: {sem:.3f} µm sem refino -> "
        f"{com:.3f} µm com refino"
    )

    checar(com < sem, f"o refino piorou o erro ({sem:.3f} -> {com:.3f} µm)")
    checar(
        com < PASSO_20X / 3,
        f"erro médio de {com:.3f} µm ainda alto para passo {PASSO_20X}",
    )


def testar_descarte_na_borda():
    """Foco real fora do alcance da busca: tem que ser recusado, e não
    devolvido como 'o melhor que achei'."""

    print("\n5. descarte quando o foco está fora da faixa")

    z_real = 100.0
    microscopio = MicroscopioFalso(plano(z0=z_real))

    # 25 µm de erro contra uma faixa de +-10: o pico não está lá dentro.
    resultado = buscar_foco(microscopio, "z", "camera", z_real + 25.0, FAIXA, PASSO)

    print(
        f"    confiável={resultado['confiavel']} motivo={resultado['motivo']!r} "
        f"({resultado['fotos']} fotos)"
    )

    checar(not resultado["confiavel"], "aceitou uma medida com o foco fora da faixa")
    checar(resultado["motivo"] == "borda", f"motivo errado: {resultado['motivo']!r}")
    # Ela varre a faixa toda de um lado antes de desistir -- é o custo
    # legítimo de ter procurado.
    checar(
        resultado["fotos"] <= FOTOS_DA_VARREDURA_COMPLETA,
        f"gastou {resultado['fotos']} fotos, mais que a varredura completa",
    )


def testar_descarte_de_pico_chato():
    """Campo cuja nitidez quase não responde ao Z. O portão de textura não
    pega esse caso (há estrutura na imagem), e o argmax de uma curva chata
    é ruído -- que viraria ponto de calibração."""

    print("\n6. descarte de curva sem pico")

    class SemResposta(MicroscopioFalso):
        def snap_image(self):
            self.fotos += 1

            # Sempre a mesma imagem, com um chiado de 0,1% para não haver
            # empate exato: é o caso limite, uma curva puramente ruidosa.
            gerador = np.random.default_rng(self.fotos)
            ruido = gerador.normal(0.0, 0.25, (LADO, LADO))
            imagem = np.clip(self.base + ruido, 0, 255).astype(np.uint8)

            bgra = np.zeros((LADO, LADO, 4), dtype=np.uint8)
            bgra[:, :, 0] = imagem
            bgra[:, :, 1] = imagem
            bgra[:, :, 2] = imagem

            self._ultimo = bgra.reshape(-1)

    microscopio = SemResposta(plano(z0=100.0))
    resultado = buscar_foco(microscopio, "z", "camera", 100.0, FAIXA, PASSO)

    print(
        f"    confiável={resultado['confiavel']} motivo={resultado['motivo']!r} "
        f"queda={resultado['queda']:.2%} (esperada {resultado['queda_esperada']:.2%})"
    )

    checar(not resultado["confiavel"], "aceitou uma medida sobre curva sem pico")
    checar(
        resultado["motivo"] == "pico_chato",
        f"motivo errado: {resultado['motivo']!r}",
    )


# =====================================================================
#  2. A ORDEM DE VISITA
# =====================================================================


def testar_serpentina():
    """A serpentina tem que visitar a grade inteira, uma vez cada, e nunca
    dar um salto maior que um campo."""

    print("\n7. ordem de visita em serpentina")

    for qtd_x, qtd_y in ((1, 1), (3, 4), (5, 5), (7, 2), (2, 9)):
        visita = list(ordem_serpentina(qtd_x, qtd_y))

        completa = set(visita) == {(i, j) for i in range(qtd_x) for j in range(qtd_y)}
        sem_repetir = len(visita) == len(set(visita))

        saltos = [
            abs(visita[k + 1][0] - visita[k][0]) + abs(visita[k + 1][1] - visita[k][1])
            for k in range(len(visita) - 1)
        ]
        salto_maximo = max(saltos) if saltos else 0

        # O raster faria, a cada troca de coluna, um salto de qtd_y - 1.
        salto_do_raster = max(qtd_y - 1, 1)

        print(
            f"    {qtd_x}x{qtd_y}: {len(visita)} posições, salto máximo "
            f"{salto_maximo} campo(s) (raster faria {salto_do_raster})"
        )

        checar(completa, f"{qtd_x}x{qtd_y}: a serpentina não cobriu a grade inteira")
        checar(sem_repetir, f"{qtd_x}x{qtd_y}: a serpentina repetiu posições")
        checar(salto_maximo <= 1, f"{qtd_x}x{qtd_y}: salto de {salto_maximo} campos")


# =====================================================================
#  3. ROBUSTEZ DO MODELO
# =====================================================================


def testar_pesos_por_idade():
    """Os pesos: monótonos, com piso para as âncoras, e inertes enquanto
    só existem âncoras."""

    print("\n8. pesos por idade")

    checar(pesos_por_idade(10, 10) is None, "envelheceu pontos sem haver medidas novas")
    checar(pesos_por_idade(3, 10) is None, "envelheceu com menos pontos que âncoras")

    pesos = pesos_por_idade(40, 10)
    presos = pesos_por_idade(40, 10, segurar_ancoras=True)

    ancoras = pesos[:10]
    medidos = pesos[10:]

    print(
        f"    40 pontos (10 âncoras): âncoras entre {ancoras.min():.3f} e "
        f"{ancoras.max():.3f}, medidos entre {medidos.min():.2f} e {medidos.max():.2f}"
    )
    print(
        f"    com segurar_ancoras: âncoras entre {presos[:10].min():.2f} e "
        f"{presos[:10].max():.2f} (piso {PESO_MINIMO_ANCORA})"
    )

    # Por padrão as âncoras envelhecem como todo mundo -- é o que deixa o
    # modelo seguir a deriva em vez de ficar preso à altura inicial.
    checar(
        ancoras.min() < PESO_MINIMO_ANCORA,
        "por padrão as âncoras não deveriam ter piso de peso",
    )
    checar(
        np.all(presos[:10] >= PESO_MINIMO_ANCORA - 1e-9),
        "com segurar_ancoras alguma âncora ficou abaixo do piso",
    )
    checar(abs(pesos[-1] - 1.0) < 1e-9, "o ponto mais recente não vale 1")
    checar(
        np.all(np.diff(medidos) > 0),
        "os pesos dos pontos medidos não crescem com a atualidade",
    )

    # Meia-vida: o ponto com MEIA_VIDA_PONTOS de idade vale metade do mais
    # recente. É a definição da constante, e é o que dá sentido ao número.
    idade_meia_vida = pesos[-1 - int(MEIA_VIDA_PONTOS)]

    print(f"    peso com {MEIA_VIDA_PONTOS:.0f} pontos de idade: {idade_meia_vida:.3f}")
    checar(abs(idade_meia_vida - 0.5) < 1e-9, "a meia-vida não bate com a constante")


def testar_modelo_segue_deriva():
    """O ponto da frente 4b: com deriva térmica, o ajuste com peso igual
    fica para trás, porque o peso de cada ponto novo é 1/N e N só cresce.
    Envelhecendo os pontos, o modelo acompanha."""

    print("\n9. o modelo acompanha a deriva térmica")

    DERIVA_POR_PONTO = 0.20  # µm de deriva a cada checkpoint

    gerador = np.random.default_rng(7)

    # 10 âncoras espalhadas, coletadas no instante 0.
    ancoras = [
        (float(x), float(y), 100.0 + 0.001 * x + 0.002 * y)
        for x in (0.0, 500.0, 1000.0)
        for y in (0.0, 500.0, 1000.0)
    ]
    ancoras.append((250.0, 750.0, 100.0 + 0.001 * 250.0 + 0.002 * 750.0))

    pontos = list(ancoras)
    qtd_ancoras = len(pontos)

    erros_com_peso = []
    erros_sem_peso = []

    # 40 checkpoints ao longo de um percurso em serpentina pela amostra,
    # com o foco derivando no tempo.
    for passo, (i, j) in enumerate(ordem_serpentina(8, 5)):
        x = 140.0 * i
        y = 250.0 * j

        deriva = DERIVA_POR_PONTO * passo
        z_verdadeiro = 100.0 + 0.001 * x + 0.002 * y + deriva

        modelo_com = reajustar_modelo_foco(pontos, "plano", qtd_ancoras=qtd_ancoras)
        modelo_sem = reajustar_modelo_foco(pontos, "plano")

        if modelo_com is not None and passo > 12:
            erros_com_peso.append(abs(calcular_z(x, y, modelo_com) - z_verdadeiro))
            erros_sem_peso.append(abs(calcular_z(x, y, modelo_sem) - z_verdadeiro))

        # O autofoco mede o Z verdadeiro com um ruidinho de medida.
        pontos.append((x, y, z_verdadeiro + gerador.normal(0.0, 0.05)))

    com = float(np.mean(erros_com_peso))
    sem = float(np.mean(erros_sem_peso))

    print(
        f"    deriva de {DERIVA_POR_PONTO} µm/checkpoint | erro médio do Z previsto: "
        f"{sem:.2f} µm com peso igual -> {com:.2f} µm com envelhecimento "
        f"({1 - com / sem:.0%} melhor)"
    )

    # O critério é a MELHORA relativa, não um erro absoluto: com deriva
    # contínua nenhum ajuste por mínimos quadrados chega a zero -- ele
    # sempre persegue um alvo que está andando, e o atraso residual é da
    # ordem de (idade média dos pontos) x (deriva por ponto). O que se
    # testa aqui é que o envelhecimento encurta esse atraso.
    checar(com < 0.8 * sem, f"o envelhecimento pouco ajudou ({sem:.2f} -> {com:.2f} µm)")


def testar_desvio_robusto():
    """O MAD tem que ignorar o outlier que o desvio padrão engoliria."""

    print("\n10. desvio robusto (MAD)")

    limpos = [0.1, -0.2, 0.05, -0.1, 0.15, -0.05]
    sujos = limpos + [30.0]

    robusto_limpo = desvio_robusto(limpos)
    robusto_sujo = desvio_robusto(sujos)
    classico_sujo = float(np.std(sujos))

    print(
        f"    sem outlier: MAD {robusto_limpo:.3f} | com outlier de 30 µm: "
        f"MAD {robusto_sujo:.3f}, desvio padrão {classico_sujo:.3f}"
    )

    checar(
        robusto_sujo < 3 * robusto_limpo,
        f"o MAD foi arrastado pelo outlier ({robusto_limpo:.3f} -> {robusto_sujo:.3f})",
    )
    checar(
        classico_sujo > 10 * robusto_sujo,
        "o desvio padrão deveria estar muito acima do MAD com um outlier de 30 µm",
    )


def testar_rejeicao_de_outlier():
    """O caminho completo, com um detrito plantado: numa posição a amostra
    'foca' 25 µm acima do plano, como se o autofoco tivesse travado em
    sujeira alta. Esse ponto não pode entrar no modelo."""

    print("\n11. medida espúria não entra no modelo")

    def superficie_com_detrito(x, y):
        z = 100.0 + 0.001 * x + 0.002 * y

        # O detrito ocupa uma posição só da grade.
        if abs(x - 500.0) < 1.0 and abs(y - 500.0) < 1.0:
            return z + 25.0

        return z

    microscopio = MicroscopioFalso(superficie_com_detrito)

    pontos = [
        (float(x), float(y), superficie_com_detrito(x, y))
        for x in (0.0, 400.0, 900.0)
        for y in (0.0, 400.0, 900.0)
    ]
    qtd_ancoras = len(pontos)

    modelo = calibrar_plano_foco(pontos, verboso=False)

    config = {
        # Faixa larga o bastante para ALCANÇAR o detrito: sem isso quem
        # rejeitaria a medida seria o guard de borda, e o teste não estaria
        # testando o filtro de outlier.
        "faixa_um": 30.0,
        "passo_um": 3.0,
        "textura_minima": 0.0,
    }

    microscopio.set_xy_position("xy", 500.0, 500.0)

    z_previsto = calcular_z(500.0, 500.0, modelo)
    microscopio.set_position("z", z_previsto)
    microscopio.snap_image()

    bgra = microscopio.get_image().reshape((LADO, LADO, 4))
    imagem = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])

    _, z_final, modelo_depois, pontos_depois, _, registro = checkpoint_autofoco(
        microscopio,
        "z",
        "camera",
        imagem,
        500.0,
        500.0,
        z_previsto,
        modelo,
        pontos,
        0.0,
        config,
        qtd_ancoras=qtd_ancoras,
    )

    print(
        f"    ação={registro['acao']!r} | correção medida "
        f"{registro['correcao']:+.1f} µm | pontos no modelo: "
        f"{len(pontos)} -> {len(pontos_depois)}"
    )

    checar(
        registro["acao"] == "descartado_outlier",
        f"a medida no detrito não foi descartada (ação {registro['acao']!r})",
    )
    checar(len(pontos_depois) == len(pontos), "o ponto espúrio entrou na lista")
    checar(modelo_depois is modelo, "o modelo foi re-ajustado com a medida espúria")
    checar(
        abs(z_final - z_previsto) < 1e-9,
        "o Z não voltou para o previsto depois do descarte",
    )


def testar_correcao_legitima_passa():
    """O contraponto do teste anterior: uma correção honesta, do tamanho
    do que o modelo erra, tem que ser aceita. Um filtro que rejeita tudo
    passaria no teste 11 e seria inútil."""

    print("\n12. correção legítima continua sendo aceita")

    z_real = 100.0
    ERRO = 2.5  # µm -- dentro da faixa e da escala do erro de modelo

    microscopio = MicroscopioFalso(plano(z0=z_real))

    pontos = [
        (float(x), float(y), z_real)
        for x in (0.0, 400.0, 900.0)
        for y in (0.0, 400.0, 900.0)
    ]
    qtd_ancoras = len(pontos)

    # Um modelo deslocado de ERRO: o autofoco tem que perceber e corrigir.
    modelo = calibrar_plano_foco(
        [(x, y, z + ERRO) for x, y, z in pontos], verboso=False
    )

    config = {"faixa_um": FAIXA, "passo_um": PASSO, "textura_minima": 0.0}

    microscopio.set_xy_position("xy", 450.0, 450.0)

    z_previsto = calcular_z(450.0, 450.0, modelo)
    microscopio.set_position("z", z_previsto)
    microscopio.snap_image()

    bgra = microscopio.get_image().reshape((LADO, LADO, 4))
    imagem = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])

    _, z_final, modelo_depois, pontos_depois, _, registro = checkpoint_autofoco(
        microscopio,
        "z",
        "camera",
        imagem,
        450.0,
        450.0,
        z_previsto,
        modelo,
        pontos,
        0.0,
        config,
        qtd_ancoras=qtd_ancoras,
    )

    print(
        f"    ação={registro['acao']!r} | correção {registro['correcao']:+.2f} µm "
        f"(erro plantado {-ERRO:+.2f}) | {registro['fotos']} fotos | "
        f"Z final {z_final:.2f} (real {z_real:.2f})"
    )

    checar(
        registro["acao"] == "modelo_reajustado",
        f"a correção legítima não foi usada (ação {registro['acao']!r})",
    )
    checar(len(pontos_depois) == len(pontos) + 1, "o ponto bom não entrou na lista")
    checar(abs(z_final - z_real) < 0.5, f"Z final errou por {z_final - z_real:+.2f} µm")


def _sequencia_de_checkpoints(deriva_por_checkpoint, com_memoria, quantidade=10):
    """Roda uma sequência de checkpoints sobre uma amostra cujo foco anda
    com o TEMPO, e devolve (ações, erro final de foco).

    `com_memoria=False` passa estado=None em cada chamada, o que faz cada
    checkpoint decidir no vácuo -- exatamente o filtro que julga a correção
    contra zero, em vez de contra a última aceita."""

    relogio = {"deriva": 0.0}

    def superficie(x, y):
        return 100.0 + 0.001 * x + 0.002 * y + relogio["deriva"]

    microscopio = MicroscopioFalso(superficie)

    pontos = [
        (float(x), float(y), superficie(x, y))
        for x in (0.0, 500.0, 1000.0)
        for y in (0.0, 500.0, 1000.0)
    ]
    qtd_ancoras = len(pontos)

    modelo = calibrar_plano_foco(pontos, verboso=False)
    config = {"faixa_um": FAIXA, "passo_um": PASSO, "textura_minima": 0.0}

    estado = {}
    acoes = []
    erro_final = 0.0

    for k in range(quantidade):
        relogio["deriva"] += deriva_por_checkpoint

        x = 100.0 * (k % 9)
        y = 120.0 * (k % 7)

        z_previsto = calcular_z(x, y, modelo)
        erro_final = z_previsto - superficie(x, y)

        microscopio.set_xy_position("xy", x, y)
        microscopio.set_position("z", z_previsto)
        microscopio.snap_image()

        bgra = microscopio.get_image().reshape((LADO, LADO, 4))
        imagem = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])

        _, _, modelo, pontos, _, registro = checkpoint_autofoco(
            microscopio, "z", "camera", imagem, x, y, z_previsto,
            modelo, pontos, 0.0, config,
            qtd_ancoras=qtd_ancoras,
            estado=estado if com_memoria else None,
        )

        acoes.append(registro["acao"])

    return acoes, erro_final


def testar_deriva_nao_vira_outlier():
    """Correções que crescem devagar, sempre na mesma direção, são DERIVA
    -- e não detrito. Um filtro que as medisse contra zero começaria a
    recusá-las assim que a deriva acumulada passasse do limite; o modelo
    pararia de ser corrigido, a deriva continuaria, e a correção seguinte
    seria maior ainda. É a espiral que este teste existe para prender."""

    print("\n14. deriva contínua não é confundida com medida espúria")

    print(
        f"    {'deriva/cp':>10}{'outliers':>10}{'bordas':>8}"
        f"{'erro no fim':>14}    (sem memória)"
    )

    for deriva in (0.3, 0.8, 2.0):
        acoes, erro = _sequencia_de_checkpoints(deriva, com_memoria=True)
        acoes_sem, erro_sem = _sequencia_de_checkpoints(deriva, com_memoria=False)

        outliers = acoes.count("descartado_outlier")
        bordas = acoes.count("descartado_na_borda")
        outliers_sem = acoes_sem.count("descartado_outlier")

        print(
            f"    {deriva:>8} µm{outliers:>10}{bordas:>8}{erro:>11.2f} µm"
            f"    {outliers_sem} outliers, {erro_sem:+.2f} µm"
        )

        # O que este teste prende: NENHUMA correção de deriva pode ser
        # recusada como espúria, em nenhum regime. Bater na borda é outra
        # coisa -- ali o foco de fato saiu do alcance da busca, e para isso
        # existe a recuperação (teste 15).
        checar(
            outliers == 0,
            f"deriva de {deriva} µm/cp: {outliers} correção(ões) recusada(s) "
            "como medida espúria",
        )
        checar(
            outliers_sem >= outliers,
            f"deriva de {deriva} µm/cp: a memória não ajudou a evitar recusas",
        )

        # O atraso do modelo é proporcional à deriva -- é o preço de um
        # estimador com memória, não uma falha. O que se cobra é que ele
        # continue dentro da faixa de busca, que é o que permite ao
        # checkpoint seguinte encontrar o foco.
        if deriva <= 0.8:
            checar(
                abs(erro) < FAIXA,
                f"deriva de {deriva} µm/cp: erro de {erro:+.2f} µm saiu da faixa",
            )

    print(
        "    (o erro cresce ~7,6 x a deriva por checkpoint: é o atraso de um\n"
        "     ajuste com meia-vida de 12 pontos. Para manter o erro abaixo de\n"
        "     meia DOF é preciso deriva < ~0,26 µm por checkpoint.)"
    )


def testar_recuperacao_apos_borda():
    """Bater na borda significa que o foco está FORA do alcance da busca.
    Sem alargar a faixa no checkpoint seguinte, ele procuraria no mesmo
    lugar e não acharia de novo -- a varredura seguiria borrada até o fim
    sem nenhuma forma de voltar."""

    print("\n15. recuperação depois de o foco sair da faixa")

    # 18 µm de erro: fora da faixa normal (+-10), dentro da alargada (+-30).
    ERRO_GRANDE = 18.0

    microscopio = MicroscopioFalso(plano(z0=100.0))

    pontos = [
        (float(x), float(y), 100.0 + ERRO_GRANDE)
        for x in (0.0, 500.0, 1000.0)
        for y in (0.0, 500.0, 1000.0)
    ]
    modelo = calibrar_plano_foco(pontos, verboso=False)

    config = {"faixa_um": FAIXA, "passo_um": PASSO, "textura_minima": 0.0}
    estado = {}
    acoes = []

    for k in range(2):
        x, y = 400.0 + 50.0 * k, 400.0
        z_previsto = calcular_z(x, y, modelo)

        microscopio.set_xy_position("xy", x, y)
        microscopio.set_position("z", z_previsto)
        microscopio.snap_image()

        bgra = microscopio.get_image().reshape((LADO, LADO, 4))
        imagem = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])

        _, _, modelo, pontos, _, registro = checkpoint_autofoco(
            microscopio, "z", "camera", imagem, x, y, z_previsto,
            modelo, pontos, 0.0, config, qtd_ancoras=9, estado=estado,
        )

        acoes.append(registro["acao"])

    print(f"    1º checkpoint: {acoes[0]} | 2º checkpoint: {acoes[1]}")

    checar(
        acoes[0] == "descartado_na_borda",
        f"o 1º checkpoint devia bater na borda, deu {acoes[0]!r}",
    )
    checar(
        acoes[1] == "modelo_reajustado",
        f"o 2º checkpoint não recuperou o foco com a faixa alargada: {acoes[1]!r}",
    )


def testar_portao_de_textura_ainda_manda():
    """Campo vazio continua sendo pulado antes de qualquer foto ser
    gasta -- a lógica que já existia não pode ter sido perdida no caminho."""

    print("\n13. o portão de textura continua valendo")

    microscopio = MicroscopioFalso(plano(z0=100.0))
    microscopio.zerar_contagem()

    # Campo uniforme: textura praticamente nula.
    vazio = np.full((LADO, LADO, 3), 128, dtype=np.uint8)

    pontos = [(0.0, 0.0, 100.0), (500.0, 0.0, 100.0), (0.0, 500.0, 100.0)]
    modelo = calibrar_plano_foco(pontos, verboso=False)

    config = {"faixa_um": FAIXA, "passo_um": PASSO, "textura_minima": 1.0}

    _, z_final, _, pontos_depois, _, registro = checkpoint_autofoco(
        microscopio, "z", "camera", vazio, 250.0, 250.0, 100.0,
        modelo, pontos, 0.0, config, qtd_ancoras=3,
    )

    print(
        f"    textura medida {medir_textura(vazio):.3f} | ação={registro['acao']!r} | "
        f"fotos gastas: {microscopio.fotos}"
    )

    checar(registro["acao"] == "pulado_sem_textura", "não pulou o campo vazio")
    checar(microscopio.fotos == 0, "gastou fotos num campo sem textura")
    checar(len(pontos_depois) == len(pontos), "o campo vazio virou ponto de calibração")


def main():
    print("=" * 68)
    print("TESTES DE AUTOFOCO (sem microscópio)")
    print("=" * 68)

    testar_forma_da_curva()
    testar_busca_acha_o_pico()
    testar_custo_quando_o_modelo_acerta()
    testar_refino_parabolico()
    testar_descarte_na_borda()
    testar_descarte_de_pico_chato()
    testar_serpentina()
    testar_pesos_por_idade()
    testar_modelo_segue_deriva()
    testar_desvio_robusto()
    testar_rejeicao_de_outlier()
    testar_correcao_legitima_passa()
    testar_deriva_nao_vira_outlier()
    testar_recuperacao_apos_borda()
    testar_portao_de_textura_ainda_manda()

    print()
    print("=" * 68)

    if falhas:
        print(f"RESULTADO: {len(falhas)} falha(s)")

        for mensagem in falhas:
            print(f"  - {mensagem}")

        print("=" * 68)
        sys.exit(1)

    print("RESULTADO: tudo passou")
    print("=" * 68)


if __name__ == "__main__":
    main()
