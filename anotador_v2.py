"""
anotador_v2.py  --  PASSO 2 do ajuste de parametros: o gabarito.

O que mudou em relacao ao anotador.py
--------------------------------------
1) AMOSTRAGEM EM VEZ DE TUDO.
   Rotular os ~30 mil candidatos da fatia levaria dezenas de horas. Em vez
   disso sorteamos uma cota por amostra. Para que a conta feche depois, o
   sorteio e ESTRATIFICADO: os candidatos sao divididos em grupos e sorteamos
   um tanto de cada grupo, guardando quantos existiam e quantos vimos. Esse
   par de numeros (o "peso" de cada rotulo) e o que permite o sweep falar da
   populacao inteira a partir de uma amostra pequena -- sem ele, olhar mais
   candidatos de score alto faria parecer que a populacao inteira e assim.

   Os grupos cruzam duas coisas:
     - a faixa de score (baixo / medio / alto);
     - se o candidato SOBREVIVE ao veto do classificador no limiar de hoje.
   O segundo eixo existe porque queremos justamente decidir se vale afrouxar
   o veto -- e para isso e preciso ter rotulo dos candidatos que ele mata.
   Sortear so entre os sobreviventes tornaria essa pergunta irrespondivel.

2) ROTULO GRADUADO, NAO BINARIO.
   Em agosto, tres regras diferentes "venceram" no F1 jogando fora justamente
   os melhores flakes -- porque eram minoria e a metrica nao sabia que valiam
   mais. A regua agora e operacional:

       2 = eu me levantaria da cadeira para ir olhar isso no microscopio
       1 = e flake, mas nao vale a viagem
       0 = nao e flake

3) TIPO DO FALSO POSITIVO.
   Quando o lixo tem cara conhecida, uma tecla so ja registra qual:
   cola, bulk (ou flake sobre bulk) ou campo vazio. Depois da para perguntar
   se cada tipo tem assinatura de cor propria -- e, principalmente, se algum
   deles e imune a qualquer limiar.

4) SEM O SCORE NA TELA.
   O v1 mostrava o score no titulo. Isso contamina: sabendo que o modelo deu
   nota alta, a pessoa hesita menos em chamar de flake, e o gabarito passa a
   concordar com a variavel que a gente queria avaliar. Aqui a tela mostra so
   a imagem -- e os candidatos vem embaralhados, para nao dar para inferir
   nada da ordem.

Teclas:
    2 = bom flake       1 = flake marginal
    0 = lixo            c = cola        b = bulk        v = vazio/substrato
    d = duvida          u = desfazer    q = salvar e sair (ESC tambem)

Uso:
    python anotador_v2.py <caminho/para/candidatos.csv> [--cota 220]
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np

# --- VISUAL ---
MARGEM_CROP = 40      # pixels de substrato mostrados ao redor do candidato
ESPESSURA = 1         # contorno fino: nao "engole" a borda do flake
COR = (0, 0, 255)     # BGR: vermelho
ALVO_MIN = 420        # amplia recortes pequenos
ALVO_MAX = 820        # reduz recortes grandes

# --- AMOSTRAGEM ---
COTA_PADRAO = 220     # candidatos rotulados por amostra
FAIXAS_SCORE = [0.05, 0.20]   # cortes entre baixo/medio/alto
LIMIAR_VETO = "cls_30"        # coluna usada para o eixo "sobrevive ao veto"
SEMENTE = 7

TECLAS = {
    ord("2"): ("2", ""),          # bom flake
    ord("1"): ("1", ""),          # flake marginal
    ord("0"): ("0", ""),          # lixo, tipo nao identificado
    ord("c"): ("0", "cola"),
    ord("b"): ("0", "bulk"),
    ord("v"): ("0", "vazio"),
    ord("d"): ("?", ""),          # duvida
}


def imread_unicode(caminho):
    return cv2.imdecode(np.fromfile(caminho, dtype=np.uint8), cv2.IMREAD_COLOR)


def estrato_de(linha):
    """Nome do grupo a que o candidato pertence: faixa de score x veto.

    O nome vai para o CSV porque o peso do rotulo depende dele -- e o sweep
    precisa saber de qual grupo cada rotulo veio para reconstruir a populacao.
    """
    s = float(linha["score"])
    faixa = "baixo" if s < FAIXAS_SCORE[0] else ("medio" if s < FAIXAS_SCORE[1] else "alto")
    veto = "vivo" if linha[LIMIAR_VETO] != "0" else "vetado"
    return f"{faixa}|{veto}"


def sortear_amostragem(linhas, cota):
    """Sorteia a cota por amostra, dividida igualmente entre os estratos.

    Divisao IGUAL, nao proporcional: os grupos pequenos (score alto, por
    exemplo) sao onde moram os flakes, e uma divisao proporcional quase nao
    olharia para eles. A distorcao que isso cria e desfeita depois pelo peso
    N_grupo/n_vistos -- e por isso que o peso e gravado agora, e nao estimado
    depois.
    """
    rng = random.Random(SEMENTE)
    por_amostra = {}
    for linha in linhas:
        por_amostra.setdefault(linha["amostra"], []).append(linha)

    escolhidos = []
    for amostra, grupo in sorted(por_amostra.items()):
        estratos = {}
        for linha in grupo:
            estratos.setdefault(estrato_de(linha), []).append(linha)

        # Reparte a cota entre os estratos; sobra de estrato pequeno e
        # redistribuida para os que ainda tem candidato de sobra.
        nomes = sorted(estratos)
        restante, pendentes = cota, list(nomes)
        alocado = {n: 0 for n in nomes}
        while restante > 0 and pendentes:
            fatia = max(1, restante // len(pendentes))
            for nome in list(pendentes):
                disponivel = len(estratos[nome]) - alocado[nome]
                leva = min(fatia, disponivel, restante)
                alocado[nome] += leva
                restante -= leva
                if alocado[nome] >= len(estratos[nome]):
                    pendentes.remove(nome)
                if restante <= 0:
                    break

        for nome in nomes:
            n = alocado[nome]
            if n == 0:
                continue
            for linha in rng.sample(estratos[nome], n):
                escolhidos.append({
                    "linha": linha,
                    "estrato": nome,
                    "n_estrato": len(estratos[nome]),   # quantos existem
                    "n_vistos": n,                      # quantos vamos olhar
                })
        vistos = sum(alocado.values())
        print(f"  {amostra:<32} {vistos:>4} de {len(grupo)} candidatos "
              f"({len(nomes)} estratos)")

    rng.shuffle(escolhidos)   # embaralha para nao dar pista pela ordem
    return escolhidos


def preparar_painel(imagem, linha):
    """Monta o painel: recorte COM contorno a esquerda, SEM contorno a direita.

    O contorno mostra o que o modelo segmentou, mas uma linha vermelha em cima
    da borda esconde exatamente a informacao que decide o rotulo -- o contraste
    da borda contra o substrato. Ver os dois lado a lado resolve isso sem
    precisar de tecla para alternar.
    """
    H, W = imagem.shape[:2]
    x, y = int(linha["x"]), int(linha["y"])
    w, h = int(linha["w"]), int(linha["h"])

    x0, y0 = max(x - MARGEM_CROP, 0), max(y - MARGEM_CROP, 0)
    x1, y1 = min(x + w + MARGEM_CROP, W), min(y + h + MARGEM_CROP, H)
    crop = imagem[y0:y1, x0:x1].copy()

    lado = max(crop.shape[0], crop.shape[1])
    if lado < ALVO_MIN:
        escala = float(ALVO_MIN // lado + 1)      # fator inteiro: pixel real
        crop = cv2.resize(crop, None, fx=escala, fy=escala,
                          interpolation=cv2.INTER_NEAREST)
    elif lado > ALVO_MAX:
        escala = ALVO_MAX / lado
        crop = cv2.resize(crop, None, fx=escala, fy=escala,
                          interpolation=cv2.INTER_AREA)
    else:
        escala = 1.0

    limpo = crop.copy()
    nums = linha["contorno"].split()
    if len(nums) >= 4:
        pts = np.array(nums, dtype=np.int32).reshape(-1, 2)
        p = ((pts - np.array([x0, y0])) * escala).astype(np.int32)
        cv2.polylines(crop, [p], isClosed=True, color=COR, thickness=ESPESSURA)

    separador = np.full((crop.shape[0], 6, 3), 40, dtype=np.uint8)
    return np.hstack([crop, separador, limpo])


def carregar_rotulos(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["cand_id"]: r for r in csv.DictReader(f)}


def salvar_rotulos(path, registros):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cand_id", "amostra", "estrato", "n_estrato", "n_vistos",
                    "rotulo", "tipo"])
        for r in registros.values():
            w.writerow([r["cand_id"], r["amostra"], r["estrato"],
                        r["n_estrato"], r["n_vistos"], r["rotulo"], r["tipo"]])


def main(csv_path, cota):
    with open(csv_path, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    print(f"candidatos no CSV: {len(linhas)}\n")
    print("amostragem estratificada (score x veto):")
    fila = sortear_amostragem(linhas, cota)

    rotulos_path = os.path.join(os.path.dirname(csv_path), "rotulos.csv")
    registros = carregar_rotulos(rotulos_path)

    pendentes = [it for it in fila if it["linha"]["cand_id"] not in registros]
    print(f"\ntotal sorteado: {len(fila)} | ja rotulados: {len(registros)} | "
          f"faltam: {len(pendentes)}")
    print("teclas: 2=bom  1=marginal  0=lixo  c=cola  b=bulk  v=vazio  "
          "d=duvida  u=undo  q=sair\n")

    cv2.namedWindow("anotador", cv2.WINDOW_NORMAL)
    cache_nome, cache_img = None, None
    historico = []

    i = 0
    while i < len(pendentes):
        item = pendentes[i]
        linha = item["linha"]
        if linha["cand_id"] in registros:
            i += 1
            continue

        if linha["imagem"] != cache_nome:
            cache_img = imread_unicode(linha["imagem"])
            cache_nome = linha["imagem"]
        if cache_img is None:
            print(f"[erro] nao li {linha['imagem']}, pulando")
            i += 1
            continue

        painel = preparar_painel(cache_img, linha)
        cv2.imshow("anotador", painel)
        cv2.setWindowTitle(
            "anotador",
            f"{len(registros) + 1}/{len(fila)}  |  {linha['amostra']}  |  "
            "2=bom 1=marginal 0=lixo c=cola b=bulk v=vazio d=? u=undo q=sair")

        tecla = cv2.waitKey(0) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("u"):
            if historico:
                ultimo = historico.pop()
                registros.pop(ultimo, None)
                salvar_rotulos(rotulos_path, registros)
                for j, it in enumerate(pendentes):
                    if it["linha"]["cand_id"] == ultimo:
                        i = j
                        break
            continue
        if tecla in TECLAS:
            rotulo, tipo = TECLAS[tecla]
            registros[linha["cand_id"]] = {
                "cand_id": linha["cand_id"],
                "amostra": linha["amostra"],
                "estrato": item["estrato"],
                "n_estrato": item["n_estrato"],
                "n_vistos": item["n_vistos"],
                "rotulo": rotulo,
                "tipo": tipo,
            }
            historico.append(linha["cand_id"])
            salvar_rotulos(rotulos_path, registros)   # salva a cada clique
            i += 1
        # qualquer outra tecla: reexibe o mesmo candidato

    cv2.destroyAllWindows()
    print(f"\nsalvo em: {rotulos_path}")
    print(f"total rotulado: {len(registros)} de {len(fila)}")
    contagem = {}
    for r in registros.values():
        chave = r["rotulo"] + (f" ({r['tipo']})" if r["tipo"] else "")
        contagem[chave] = contagem.get(chave, 0) + 1
    for chave in sorted(contagem):
        print(f"  {chave:<12} {contagem[chave]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="Caminho para o candidatos.csv do extrator v2")
    p.add_argument("--cota", type=int, default=COTA_PADRAO,
                   help=f"Candidatos a rotular POR AMOSTRA (default: {COTA_PADRAO})")
    args = p.parse_args()
    main(os.path.abspath(args.csv), args.cota)
