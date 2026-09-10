"""
extrair_candidatos_v2.py  --  PASSO 1 do ajuste de parametros, versao multi-amostra.

O que mudou em relacao ao extrair_candidatos.py
------------------------------------------------
O v1 rodava as redes uma vez e gravava o que os TRES limiares ajustaveis
comparam (score, size, max_prob). Isso deixava de fora justamente o filtro que
mais mata candidato -- o veto do classificador -- e por isso o sweep de agosto
otimizou um pipeline que o microscopio nao executa.

Aqui a passada pela GPU e a mesma (a parte cara e desenhar as mascaras), mas
gravamos tudo que permite reproduzir offline QUALQUER decisao posterior:

1) O VETO, SEPARADO EM SUAS DUAS CAUSAS.
   O classificador (AMM) calcula, para cada pixel, a distancia ate cada uma das
   5 classes conhecidas. Fica com a mais proxima -- e depois zera ("vira fundo")
   todo pixel cuja distancia passe de um limiar. O candidato morre se a classe
   majoritaria dentro da mascara for 0. Ou seja, ha DOIS jeitos de morrer:

       (a) longe de tudo    -> os pixels nao se parecem com nada que o modelo
                               conhece; e o LIMIAR_MAHALANOBIS que decide.
       (b) parece substrato -> os pixels estao PERTO da classe 0, que e o
                               proprio substrato. Afrouxar o limiar nao ajuda:
                               eles continuam sendo classificados como fundo.

   O codigo do MaskTerial funde as duas num unico numero e a distincao se perde.
   Aqui gravamos o mapa de distancia ANTES do corte (get_mh_distance_map), o que
   permite (i) reaplicar qualquer limiar offline e (ii) medir quanto de cada
   causa existe. Se o veto for dominado por (b), afrouxar o limiar e empurrar a
   porta errada -- e melhor descobrir isso antes de rotular 1500 candidatos.

2) O CONTRASTE, DIRETO DA MASCARA DE VERDADE.
   O medir_contraste.py reconstruia a mascara de um poligono simplificado e
   relia as imagens todas de novo. Aqui o contrast_image ja esta em memoria:
   sai de graca e sem aproximacao.

3) UMA FATIA ESPALHADA, NAO A VARREDURA INTEIRA.
   Para ESCOLHER parametros nao e preciso rodar as 66 mil imagens -- uma amostra
   bem espalhada da quase a mesma resposta. O sorteio usa semente fixa: rodar de
   novo pega exatamente os mesmos tiles. Rodar tudo fica para DEPOIS da escolha,
   quando o objetivo passa a ser produzir a lista de flakes de verdade.

Saidas (na pasta --saida):
    candidatos.csv  -- uma linha por candidato
    tiles.csv       -- uma linha por imagem lida (inclusive as sem candidato:
                       sem elas nao da para calcular candidatos por tile)

Uso:
    python extrair_candidatos_v2.py <raiz> [<raiz> ...] [--tiles N]
                                    [--ignorar NOME ...] [--saida PASTA]
                                    [--continuar]
"""

import argparse
import csv
import os
import random
import time

import cv2
import numpy as np
import torch

from comandos import medir_foco_relativo
from modelo import obter_predictor
from parametros import LIMIAR_MAHALANOBIS

# Piso de score na gravacao: abaixo disso e ruido puro (~96% das propostas).
# O trabalho caro (rodar as redes) e o mesmo; o que muda e nao escrever lixo.
PISO_EXTRACAO = 0.015

# Simplificacao do contorno: fracao do perimetro usada como tolerancia.
EPSILON_FRAC = 0.01

# Limiares de Mahalanobis reproduzidos offline. O 'inf' e o pipeline SEM veto
# por distancia -- quem morrer nele morreu por parecer substrato (causa b).
GRADE_MAHALANOBIS = [10.0, 20.0, 30.0, 50.0, 100.0, float("inf")]

# 5 componentes no AMM/GrapheneH (meta_data.json: num_classes=5), sendo o
# indice 0 o substrato.
N_CLASSES = 5

# Largura, em pixels, do anel de substrato medido em volta de cada candidato.
# 12 px a 0,17 um/px sao ~2 um: perto o bastante para pegar a MESMA vizinhanca
# (mesma iluminacao, mesma sujeira de fundo) e longe o bastante para nao pegar
# a propria borda difusa do flake.
RAIO_ANEL = 12

# Semente do sorteio de tiles. Fixa de proposito: reprodutibilidade.
SEMENTE = 42


def nome_limiar(t):
    """'inf' em vez de 'inf.0' para o cabecalho do CSV ficar legivel."""
    return "inf" if t == float("inf") else f"{int(t)}"


def imread_unicode(caminho):
    """Le imagem mesmo com acento no caminho.

    O cv2.imread usa a codepage ANSI do Windows e devolve None em caminhos
    acentuados. Aqui lemos os bytes com np.fromfile -- que entende Unicode --
    e decodificamos com cv2.imdecode.
    """
    return cv2.imdecode(np.fromfile(caminho, dtype=np.uint8), cv2.IMREAD_COLOR)


def descobrir_amostras(raizes, ignorar):
    """Acha as pastas que contem .tif e devolve [(nome_amostra, [caminhos])].

    O nome carrega a raiz ("varredura 19.08/Amostra_1") porque as duas
    varreduras tem uma pasta chamada Amostra_1 -- sem o prefixo elas colidiriam
    e o sweep acharia que sao a mesma amostra.
    """
    amostras = []
    for raiz in raizes:
        raiz = os.path.abspath(raiz)
        etiqueta_raiz = os.path.basename(raiz.rstrip("\\/"))
        for pasta, _, arquivos in os.walk(raiz):
            tifs = sorted(f for f in arquivos if f.lower().endswith(".tif"))
            if not tifs:
                continue
            curto = os.path.basename(pasta)
            nome = etiqueta_raiz if pasta == raiz else f"{etiqueta_raiz}/{curto}"
            if any(termo.lower() in nome.lower() for termo in ignorar):
                print(f"[ignorada] {nome}")
                continue
            amostras.append((nome, [os.path.join(pasta, f) for f in tifs]))
    return amostras


def sortear(caminhos, n):
    """Sorteia n tiles espalhados pela amostra.

    Por que sortear e nao pegar os n primeiros: os nomes sao img_pos_X0_Y0,
    X0_Y1, ... -- os primeiros n sao uma COLUNA da amostra, nao a amostra. Se
    aquela faixa estiver fora de foco ou for so borda, a fatia inteira mente.
    """
    if n <= 0 or n >= len(caminhos):
        return caminhos
    return sorted(random.Random(SEMENTE).sample(caminhos, n))


@torch.inference_mode()
def analisar(predictor, imagem_bgr):
    """Roda as redes numa imagem e devolve (info_do_tile, [candidatos]).

    Reproduz o miolo de _inference_with_segmentation_model + infer_classes do
    MaskTerial, mas sem aplicar limiar nenhum -- e guardando o que o MaskTerial
    descarta pelo caminho.
    """
    dispositivo = predictor.device
    imagem = torch.tensor(imagem_bgr, dtype=torch.uint8).to(dispositivo)

    # --- (1) Grandezas do TILE (valem mesmo se nenhum candidato aparecer) ---
    fundo = predictor.calculate_mean_background_values(imagem).flatten().tolist()
    info_tile = {
        "fundo_b": fundo[0], "fundo_g": fundo[1], "fundo_r": fundo[2],
        "foco": medir_foco_relativo(imagem_bgr),
    }

    # --- (2) SEGMENTACAO: o passo caro. Mascaras + score por candidato. ---
    instances = predictor.segmentation_model(imagem)
    if len(instances) == 0:
        return info_tile, []

    # --- (3) CLASSIFICACAO, sem o corte por distancia. ---
    # contrast_image = imagem / cor_de_fundo - 1, por canal (BGR).
    contrast_image = predictor.calculate_contrast_image(imagem)

    tipo = dispositivo if isinstance(dispositivo, str) else dispositivo.type
    with torch.autocast(device_type=tipo, dtype=torch.float16):
        # (K, H, W): a distancia de cada pixel a CADA uma das 5 classes.
        mapa = predictor.classification_model.get_mh_distance_map(contrast_image)

    # Fica com a classe mais proxima e a distancia ate ela. E exatamente o que o
    # postprocess_mh_map faz -- so que sem zerar quem passou do limiar, que e a
    # informacao que queremos preservar.
    dist_px, classe_px = mapa.min(dim=0)
    dist_px = dist_px.float()

    candidatos = []
    for i in range(len(instances)):
        mask = instances[i].pred_masks.squeeze()
        score = instances[i].scores.squeeze().item()
        if score < PISO_EXTRACAO:
            continue

        # size = pixels da mascara crua: e o que o SIZE_THRESHOLD compara.
        size_px = int(torch.count_nonzero(mask).item())
        if size_px == 0:
            continue

        # --- forma: maior contorno externo, simplificado ---
        mask_np = (mask.cpu().numpy() > 0).astype(np.uint8)
        contornos, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            continue
        maior = max(contornos, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(maior)
        eps = EPSILON_FRAC * cv2.arcLength(maior, True)
        aprox = cv2.approxPolyDP(maior, eps, True).reshape(-1, 2)

        # Mascara PREENCHIDA do maior contorno -- e a que o MaskTerial usa para
        # o mean_contrast (extract_properties devolve ela). Difere da crua por
        # tapar buracos internos.
        cheia = np.zeros_like(mask_np)
        cv2.drawContours(cheia, [maior], -1, 1, -1)
        cheia_t = torch.from_numpy(cheia.astype(bool)).to(dispositivo)
        c_dentro = torch.mean(contrast_image[cheia_t], dim=0).float()
        contraste = c_dentro.tolist()

        # --- UNIFORMIDADE: o desvio que a media joga fora ---------------------
        # Uma monocamada tem espessura constante, logo contraste constante. Cola
        # tem espessura variavel; substrato vazio e ruido em torno de zero. As
        # tres podem ter a MESMA media com desvios completamente diferentes --
        # medido em 27/08, monolayer bom, cola e vazio caiam todos em
        # c ~ (-0,03, -0,07, -0,11) e o modelo nao tinha como separa-los.
        # torch.std devolve NaN com menos de 2 pixels -- raro (mascaras de 1px),
        # mas NaN vaza silenciosamente ate o sklearn recusar a matriz inteira.
        pixels_dentro = contrast_image[cheia_t]
        desvio = (torch.std(pixels_dentro, dim=0).float().tolist()
                  if pixels_dentro.shape[0] >= 2 else [0.0, 0.0, 0.0])

        # --- CONTRASTE LOCAL: contra o anel, nao contra o fundo global --------
        # O contrast_image compara cada pixel com o modo de brilho da imagem
        # INTEIRA. Isso confunde um flake de baixo contraste com uma variacao
        # suave de iluminacao: as duas ficam ~2% abaixo do fundo global.
        # A diferenca e que o flake tem um DEGRAU contra a vizinhanca imediata,
        # e a variacao de iluminacao nao tem. Medir contra o anel isola isso.
        kernel_anel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * RAIO_ANEL + 1, 2 * RAIO_ANEL + 1))
        anel_np = cv2.dilate(cheia, kernel_anel) - cheia
        anel_t = torch.from_numpy(anel_np.astype(bool)).to(dispositivo)
        if bool(anel_t.any()):
            c_anel = torch.mean(contrast_image[anel_t], dim=0).float()
            # contrast_image = img/fundo - 1, entao img/fundo = c + 1. A razao
            # dentro/anel e (c_in+1)/(c_anel+1); menos 1 devolve um contraste na
            # mesma escala dos outros. O clamp evita divisao por ~0 num anel
            # patologicamente escuro.
            local = (((c_dentro + 1.0) / torch.clamp(c_anel + 1.0, min=0.05))
                     - 1.0).tolist()
        else:
            local = [0.0, 0.0, 0.0]

        # --- o veto, dissecado ---
        # Usa a mascara CRUA porque e a que o infer_classes usa (sem_seg[mask]).
        d = dist_px[mask]
        c = classe_px[mask]

        # Causa (b): fracao de pixels cuja classe mais proxima E o substrato,
        # independentemente da distancia. Limiar nenhum resgata esses pixels.
        frac_substrato = float((c == 0).float().mean().item())

        # Percentis da distancia: dizem o quao longe do conhecido o candidato
        # esta, sem depender de limiar nenhum.
        q = torch.quantile(d, torch.tensor([0.25, 0.5, 0.75], device=d.device))

        # Reaplica cada limiar: pixel longe demais vira classe 0 (substrato),
        # depois a classe majoritaria decide o destino do candidato.
        por_limiar = {}
        for t in GRADE_MAHALANOBIS:
            ct = torch.where(d > t, torch.zeros_like(c), c)
            contagens = torch.bincount(ct, minlength=N_CLASSES).float()
            ocupacao = contagens / contagens.sum()
            occ, cls = torch.max(ocupacao, dim=0)
            por_limiar[nome_limiar(t)] = (int(cls.item()), float(occ.item()))

        candidatos.append({
            "score": float(score),
            "size_px": size_px,
            "area": float(cv2.contourArea(maior)),
            "x": x, "y": y, "w": w, "h": h,
            "c_b": contraste[0], "c_g": contraste[1], "c_r": contraste[2],
            "std_b": desvio[0], "std_g": desvio[1], "std_r": desvio[2],
            "cl_b": local[0], "cl_g": local[1], "cl_r": local[2],
            "d_p25": float(q[0].item()), "d_p50": float(q[1].item()),
            "d_p75": float(q[2].item()),
            "frac_substrato": frac_substrato,
            "por_limiar": por_limiar,
            "contorno": " ".join(str(int(v)) for v in aprox.flatten()),
        })

    return info_tile, candidatos


def cabecalho_candidatos():
    colunas = ["cand_id", "amostra", "imagem", "score", "size_px", "area",
               "x", "y", "w", "h", "c_b", "c_g", "c_r",
               "std_b", "std_g", "std_r", "cl_b", "cl_g", "cl_r",
               "d_p25", "d_p50", "d_p75", "frac_substrato"]
    for t in GRADE_MAHALANOBIS:
        colunas += [f"cls_{nome_limiar(t)}", f"occ_{nome_limiar(t)}"]
    colunas.append("contorno")
    return colunas


def main(raizes, n_tiles, ignorar, saida, continuar):
    amostras = descobrir_amostras(raizes, ignorar)
    if not amostras:
        print("Nenhuma pasta com .tif encontrada.")
        return

    tarefas = [(nome, sortear(caminhos, n_tiles)) for nome, caminhos in amostras]
    total_tiles = sum(len(c) for _, c in tarefas)
    print(f"\n{len(tarefas)} amostra(s), {total_tiles} tiles a processar:")
    for nome, caminhos in tarefas:
        print(f"  {nome:<40} {len(caminhos)} tiles")
    print(f"\nlimiar de Mahalanobis em uso no modelo: {LIMIAR_MAHALANOBIS}")
    print("limiares reproduzidos offline: "
          f"{[nome_limiar(t) for t in GRADE_MAHALANOBIS]}\n")

    os.makedirs(saida, exist_ok=True)
    caminho_cand = os.path.join(saida, "candidatos.csv")
    caminho_tiles = os.path.join(saida, "tiles.csv")

    # Retomada: o tiles.csv lista TODA imagem ja lida, entao ele e o registro
    # de progresso. Sem isso, uma queda no meio custaria a corrida inteira.
    ja_feitos = set()
    proximo_id = 0
    if continuar and os.path.exists(caminho_tiles):
        with open(caminho_tiles, encoding="utf-8") as f:
            ja_feitos = {r["imagem"] for r in csv.DictReader(f)}
        if os.path.exists(caminho_cand):
            with open(caminho_cand, encoding="utf-8") as f:
                ids = [int(r["cand_id"]) for r in csv.DictReader(f)]
            proximo_id = max(ids) + 1 if ids else 0
        print(f"[continuar] {len(ja_feitos)} tiles ja processados, retomando.\n")

    modo = "a" if ja_feitos else "w"
    predictor = obter_predictor()

    f_cand = open(caminho_cand, modo, newline="", encoding="utf-8")
    f_tiles = open(caminho_tiles, modo, newline="", encoding="utf-8")
    w_cand, w_tiles = csv.writer(f_cand), csv.writer(f_tiles)
    if modo == "w":
        w_cand.writerow(cabecalho_candidatos())
        w_tiles.writerow(["amostra", "imagem", "n_candidatos",
                          "fundo_b", "fundo_g", "fundo_r", "foco"])

    t0 = time.perf_counter()
    feitos, total_cand = 0, proximo_id
    try:
        for nome, caminhos in tarefas:
            for caminho in caminhos:
                if caminho in ja_feitos:
                    continue
                imagem = imread_unicode(caminho)
                if imagem is None:
                    print(f"[erro] nao li: {caminho}")
                    continue

                info, candidatos = analisar(predictor, imagem)
                w_tiles.writerow([
                    nome, caminho, len(candidatos),
                    f"{info['fundo_b']:.2f}", f"{info['fundo_g']:.2f}",
                    f"{info['fundo_r']:.2f}", f"{info['foco']:.5f}",
                ])
                for cand in candidatos:
                    linha = [
                        total_cand, nome, caminho,
                        f"{cand['score']:.6f}", cand["size_px"],
                        f"{cand['area']:.1f}",
                        cand["x"], cand["y"], cand["w"], cand["h"],
                        f"{cand['c_b']:.5f}", f"{cand['c_g']:.5f}",
                        f"{cand['c_r']:.5f}",
                        f"{cand['std_b']:.5f}", f"{cand['std_g']:.5f}",
                        f"{cand['std_r']:.5f}",
                        f"{cand['cl_b']:.5f}", f"{cand['cl_g']:.5f}",
                        f"{cand['cl_r']:.5f}",
                        f"{cand['d_p25']:.3f}", f"{cand['d_p50']:.3f}",
                        f"{cand['d_p75']:.3f}", f"{cand['frac_substrato']:.4f}",
                    ]
                    for t in GRADE_MAHALANOBIS:
                        cls, occ = cand["por_limiar"][nome_limiar(t)]
                        linha += [cls, f"{occ:.4f}"]
                    linha.append(cand["contorno"])
                    w_cand.writerow(linha)
                    total_cand += 1

                feitos += 1
                if feitos % 200 == 0:
                    f_cand.flush()
                    f_tiles.flush()
                    decorrido = time.perf_counter() - t0
                    taxa = feitos / decorrido
                    restam = (total_tiles - len(ja_feitos) - feitos) / max(taxa, 1e-9)
                    print(f"  {feitos} tiles | {taxa:.2f} img/s | "
                          f"{total_cand - proximo_id} candidatos | "
                          f"faltam ~{restam/60:.0f} min")
    except KeyboardInterrupt:
        print("\n[interrompido] o que ja foi lido esta salvo; use --continuar.")
    finally:
        f_cand.close()
        f_tiles.close()

    print(f"\nPronto. {feitos} tiles, {total_cand - proximo_id} candidatos novos.")
    print(f"  {caminho_cand}")
    print(f"  {caminho_tiles}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raizes", nargs="+",
                   help="Pasta(s) da(s) varredura(s). Subpastas com .tif viram amostras.")
    p.add_argument("--tiles", type=int, default=2500,
                   help="Quantos tiles sortear POR AMOSTRA (0 = todos). Default: 2500")
    p.add_argument("--ignorar", nargs="*", default=[],
                   help="Trechos de nome de amostra a pular (ex: Amostra_3)")
    p.add_argument("--saida", default=None,
                   help="Pasta de saida (default: <1a raiz>/avaliacao_parametros)")
    p.add_argument("--continuar", action="store_true",
                   help="Retoma de onde parou, lendo o tiles.csv ja gravado")
    args = p.parse_args()

    destino = args.saida or os.path.join(os.path.abspath(args.raizes[0]),
                                         "avaliacao_parametros")
    main(args.raizes, args.tiles, args.ignorar, destino, args.continuar)
