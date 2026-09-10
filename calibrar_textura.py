"""
calibrar_textura.py -- acha a textura_minima de uma objetiva.

O QUE É ESSE LIMIAR
-------------------
Durante a varredura, antes de rodar o autofoco num campo, o script pergunta
se aquele campo tem estrutura suficiente para se medir foco nele. Quem
responde é a medir_textura, e o limiar dessa resposta é a textura_minima.

Ele existe porque a métrica de foco (variância do Laplaciano) não distingue
"campo em foco" de "campo vazio": no substrato limpo ela mede o RUÍDO do
sensor, que não tem nada a ver com o Z. Rodar autofoco ali devolveria uma
posição aleatória -- e, pior, essa posição entraria no modelo de foco como
se fosse um ponto medido, estragando todas as posições seguintes.

POR QUE NÃO DÁ PARA COPIAR O VALOR DE OUTRA OBJETIVA
-----------------------------------------------------
A medir_textura devolve um desvio padrão em níveis de cinza de uma imagem
de 8 bits: ela ESCALA COM O BRILHO. Metade da exposição, metade da textura,
no mesmo campo. Como cada objetiva tem sua exposição, cada uma tem seu
limiar -- não é preciosismo, é fator multiplicativo direto.

É por isso que este script aplica a exposição e o balanço de branco da
objetiva escolhida ANTES do primeiro snap. Medir com outra exposição dá um
número que não serve para nada.

COMO USAR
---------
Rode DEPOIS do calibrar_camera.py (é dele que vêm exposição e balanço de
branco). Amostre alguns campos vazios e alguns com estrutura -- pelo menos
3 de cada, espalhados pela amostra. O script sugere o limiar no fim.

    python calibrar_textura.py
"""

import csv
import datetime
import math
import os

import numpy as np
from pycromanager import Core
from tifffile import imwrite

from caminhos import DIRETORIO_SAIDA_BASE
from capturar_imagem import capturar_imagem
from comandos import escolher_objetiva, medir_textura

# Este script precisa da exposição e do balanço de branco, e só. Exigir o
# fov aqui obrigaria a rodar o calibrar_fov.py antes sem nenhuma razão
# técnica.
CAMPOS_NECESSARIOS = ("exposicao", "wb_red", "wb_blue")

# Abaixo disto o resumo avisa que a amostragem foi pequena demais para
# confiar nos extremos (que são justamente o que define o limiar).
MINIMO_POR_GRUPO = 3

TECLAS = {"v": "vazio", "f": "estrutura"}


def resumir(valores):
    """Devolve (n, mínimo, mediana, máximo) de uma lista de medidas."""

    arranjo = np.array(valores, dtype=float)

    return len(arranjo), arranjo.min(), float(np.median(arranjo)), arranjo.max()


def sugerir_limiar(vazio, estrutura):
    """Limiar sugerido entre os dois grupos, e se eles se sobrepõem.

    Usa a MÉDIA GEOMÉTRICA -- raiz do produto -- e não a aritmética. A
    textura varia multiplicativamente (campo vazio ~1, campo com flake ~15:
    uma ordem de grandeza), e nesse regime a média aritmética cola no valor
    alto: entre 1 e 15 ela daria 8, perto demais do campo bom, e um campo
    com estrutura fraca seria descartado. A geométrica dá ~3,9, que fica no
    meio em PROPORÇÃO -- 3,9x acima do vazio e 3,9x abaixo do cheio.

    E não usa as medianas, e sim os EXTREMOS que se encaram: o campo vazio
    mais texturizado e o campo com estrutura mais fraco. São eles que
    definem a fronteira; as medianas dariam um limiar que erra justamente
    nos casos difíceis."""

    pior_vazio = max(vazio)
    pior_estrutura = min(estrutura)

    sobrepoem = pior_vazio >= pior_estrutura

    if pior_vazio <= 0:
        # Campo vazio com textura zero é imagem perfeitamente uniforme --
        # na prática, tampa fechada ou sensor saturado. A média geométrica
        # daria zero (limiar que nunca pula nada), então cai-se num terço
        # do campo mais fraco com estrutura.
        return pior_estrutura / 3.0, sobrepoem

    return math.sqrt(pior_vazio * pior_estrutura), sobrepoem


def main():
    core = Core()
    camera = core.get_camera_device()

    nome_objetiva, config = escolher_objetiva(exigidos=CAMPOS_NECESSARIOS)

    # O passo que faz o número valer alguma coisa. Ver o cabeçalho.
    core.set_exposure(config["exposicao"])
    core.set_property(camera, "WhiteBalanceRed", config["wb_red"])
    core.set_property(camera, "WhiteBalanceBlue", config["wb_blue"])

    print(
        f"\nExposição {config['exposicao']} ms e balanço de branco "
        f"{config['wb_red']}/{config['wb_blue']} aplicados (da objetiva {nome_objetiva})."
    )

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    diretorio = os.path.join(
        DIRETORIO_SAIDA_BASE, f"Textura_{nome_objetiva}_{timestamp}"
    )
    os.makedirs(diretorio, exist_ok=True)

    print("\n--- AMOSTRAGEM ---")
    print("Leve o estágio até um campo e classifique:")
    print("  v      -> campo VAZIO (só substrato limpo)")
    print("  f      -> campo com ESTRUTURA (flake, borda de chip, sujeira visível)")
    print("  ENTER  -> terminar e ver o resultado")
    print(
        f"\nPelo menos {MINIMO_POR_GRUPO} de cada, espalhados pela amostra. O foco não\n"
        "precisa estar perfeito: a medir_textura foi feita para quase não responder\n"
        "ao foco -- se respondesse, ela bloquearia o autofoco justamente quando a\n"
        "imagem está desfocada, que é quando ele é necessário."
    )
    print(
        "\nCampo duvidoso (sujeira fraca, borda difusa): classifique como VAZIO. Isso\n"
        "empurra o limiar para cima, e os dois erros não custam a mesma coisa --\n"
        "pular um checkpoint só perde uma chance de correção, enquanto autofocar em\n"
        "ruído injeta um ponto errado que contamina o modelo de foco inteiro."
    )

    medidas = {"vazio": [], "estrutura": []}
    linhas = []

    while True:
        escolha = input("\n[v] vazio  [f] estrutura  [ENTER] terminar: ").strip().lower()

        if escolha == "":
            break

        if escolha not in TECLAS:
            print("[Erro] Digite v, f, ou ENTER para terminar.")
            continue

        # O live é parado a cada snap, e não uma vez no começo como nos
        # outros scripts de calibração: aqui você navega entre um campo e
        # outro, e navegar sem live é sofrimento desnecessário. Pode deixar
        # ligado no Micro-Manager -- o script tira a câmera dele quando
        # precisa fotografar.
        if core.is_sequence_running():
            core.stop_sequence_acquisition()

        imagem = capturar_imagem(core, camera)
        textura = medir_textura(imagem)

        grupo = TECLAS[escolha]
        medidas[grupo].append(textura)

        nome_arquivo = f"{grupo}_{len(medidas[grupo]):02d}.tif"
        imwrite(os.path.join(diretorio, nome_arquivo), imagem)
        linhas.append([grupo, f"{textura:.4f}", nome_arquivo])

        print(
            f"  {grupo:<9} textura = {textura:7.2f}   "
            f"({len(medidas['vazio'])} vazio, {len(medidas['estrutura'])} estrutura)"
        )

    with open(os.path.join(diretorio, "texturas.csv"), "w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")
        escritor.writerow(["Grupo", "Textura", "Arquivo"])
        escritor.writerows(linhas)

    print(f"\nImagens e texturas.csv em: {diretorio}")

    # --- resultado ---
    print("\n" + "=" * 62)
    print("RESULTADO")
    print("=" * 62)

    if not medidas["vazio"] or not medidas["estrutura"]:
        print(
            "\nFaltou um dos grupos, então não há limiar a sugerir: ele é por\n"
            "definição a fronteira ENTRE os dois. As imagens ficaram salvas --\n"
            "rode de novo e amostre os dois lados."
        )
        return

    print(f"\n{'grupo':<11} {'n':>3} {'mínimo':>9} {'mediana':>9} {'máximo':>9}")

    for grupo in ("vazio", "estrutura"):
        n, minimo, mediana, maximo = resumir(medidas[grupo])
        print(f"{grupo:<11} {n:>3} {minimo:>9.2f} {mediana:>9.2f} {maximo:>9.2f}")

    limiar, sobrepoem = sugerir_limiar(medidas["vazio"], medidas["estrutura"])

    if sobrepoem:
        print(
            f"\n[Atenção] Os grupos se SOBREPÕEM: o campo vazio mais texturizado\n"
            f"          ({max(medidas['vazio']):.2f}) pontuou mais que o campo com estrutura\n"
            f"          mais fraco ({min(medidas['estrutura']):.2f}). Nenhum limiar separa os dois.\n"
            "\n"
            "          Duas explicações possíveis, e elas pedem ações opostas:\n"
            "          (a) algum campo foi classificado errado -- confira as imagens\n"
            "              salvas, é para isso que elas existem;\n"
            "          (b) nesta objetiva a textura realmente não separa os casos, e\n"
            "              aí o portão não vai funcionar: melhor deixar o limiar alto\n"
            "              e aceitar autofoco raro do que baixo e deixar entrar ruído."
        )

    for grupo in ("vazio", "estrutura"):
        if len(medidas[grupo]) < MINIMO_POR_GRUPO:
            print(
                f"\n[Aviso] Só {len(medidas[grupo])} amostra(s) de '{grupo}'. O limiar sai dos\n"
                f"        valores EXTREMOS de cada grupo, e com poucas amostras o extremo\n"
                "        que importa provavelmente nem foi visto."
            )

    print(f"\nLimiar sugerido (média geométrica dos extremos): {limiar:.2f}")
    print("\nCole no parametros.py:")
    print(f'    OBJETIVAS["{nome_objetiva}"]["textura_minima"] = {limiar:.2f}')

    print(
        "\nPara refinar depois: uma varredura de verdade grava um autofoco.csv por\n"
        "amostra, com a textura de cada checkpoint. Ele mostra a distribuição ao\n"
        "longo da amostra inteira -- incluindo os campos ambíguos que uma amostragem\n"
        "manual de dez pontos não pega."
    )


if __name__ == "__main__":
    main()
