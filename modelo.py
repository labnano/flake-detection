import os

import cv2
import joblib
from maskterial import MaskTerial, load_models
from maskterial.structures import Flake
import torch
import matplotlib.pyplot as plt
import numpy as np
from caminhos import SEG_MODEL_ROOT, CLS_MODEL_ROOT
from parametros import (
    SCORE_THRESHOLD,
    MIN_CLASS_OCCUPANCY,
    SIZE_THRESHOLD,
    LIMIAR_CONTRASTE_G,
    LIMIAR_PROB_FLAKE,
    RAIO_FUSAO_PX,
    ESPESSURA_CONTORNO,
    LIMIAR_MAHALANOBIS
)

# Modelo treinado (regressao logistica sobre contraste BGR + frac_substrato).
# Fica ao lado deste arquivo para funcionar nas duas maquinas sem editar caminho.
CAMINHO_FILTRO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "modelo_filtro_flake.joblib")

CORES_PADRAO = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 41, 255),
]


def display_results(
    image: np.ndarray,
    flakes: list[Flake],
    colors: list[tuple[int, int, int]] | None = None,
    show: bool = False,
):
    if colors is None:
        colors = CORES_PADRAO

    # Faz uma cópia para não modificar a imagem original diretamente.
    image = image.copy()

    for flake in flakes:
        mask = flake.mask.astype(np.uint8)

        # Garante que a máscara esteja no formato esperado pelo OpenCV.
        if mask.max() <= 1:
            mask = mask * 255

        class_id = int(flake.thickness)

        # Evita erro caso class_id seja maior que o número de cores disponíveis.
        color = colors[class_id % len(colors)]

        # Draw outline
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, 2)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(mask)

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # Add class label
        label = f"Class {class_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Adjust text position to keep it within bounds
        text_y = y - 5 if y - text_height - 10 >= 0 else y + h + text_height + 5
        bg_y1 = text_y - text_height - 5 if y - text_height - 10 >= 0 else y + h
        bg_y2 = text_y + 5 if y - text_height - 10 >= 0 else y + h + text_height + 10

        # Draw background rectangle for text
        cv2.rectangle(image, (x, bg_y1), (x + text_width, bg_y2), color, -1)

        # Draw text
        cv2.putText(
            image, label, (x, text_y), font, font_scale, (255, 255, 255), thickness
        )

    # Durante a varredura automática, não é bom abrir uma figura a cada flake.
    # Se quiser visualizar manualmente, chame display_results(image, flakes, show=True).
    if show:
        plt.subplots(1, 1, figsize=(12, 12), dpi=100)
        plt.imshow(image[:, :, ::-1])
        plt.axis("off")
        plt.show()

    return image


def filtrar_por_cor(flakes):
    """Pós-filtro de cor: mantém só os flakes com contraste verde suficientemente
    negativo (assinatura óptica do grafeno em SiO2) e descarta a sujeira de cola,
    que tem contraste verde ~0 ou positivo.

    Cada Flake já carrega o mean_contrast (BGR) calculado pelo MaskTerial, então
    o canal verde é o índice 1 -- não precisa recalcular nada. O limiar mora em
    parametros.py (LIMIAR_CONTRASTE_G), com o racional da escolha.
    """
    return [f for f in flakes if f.mean_contrast[1] <= LIMIAR_CONTRASTE_G]


_filtro = None


def obter_filtro():
    """Carrega (uma vez) o modelo treinado que decide flake x lixo.

    Devolve None se o arquivo nao existir -- assim uma maquina sem o .joblib
    continua rodando com o filtro de cor antigo em vez de quebrar.
    Para regerar o arquivo, veja o cabecalho de treinar_filtro.py.
    """
    global _filtro
    if _filtro is None:
        if not os.path.isfile(CAMINHO_FILTRO):
            return None
        _filtro = joblib.load(CAMINHO_FILTRO)
    return _filtro


@torch.inference_mode()
def calcular_extras(imagem_bgr, flakes):
    """Features que o objeto Flake nao traz prontas: frac_substrato e o DESVIO
    do contraste dentro da mascara.

    O desvio e o que separa monocamada de substrato vazio. Os dois tem contraste
    MEDIO quase igual -- medido em 27/08, um monolayer bom deu c = (-0,020,
    -0,064, -0,102) e um vazio deu (-0,044, -0,073, -0,115), indistinguiveis --
    mas a monocamada tem espessura constante, logo contraste CONSISTENTE, e o
    vazio e ruido do sensor. A razao |c_g|/std_g separa os dois por fator 8
    (5,8 contra 0,7, com os quartis sem se tocar).

    Devolve um dicionario de listas, uma entrada por flake, na mesma ordem.
    """
    vazio = {k: [] for k in ["frac_substrato", "std_b", "std_g", "std_r"]}
    if not flakes:
        return vazio

    predictor = obter_predictor()
    imagem = torch.tensor(imagem_bgr, dtype=torch.uint8).to(predictor.device)
    contraste = predictor.calculate_contrast_image(imagem)

    tipo = predictor.device if isinstance(predictor.device, str) else predictor.device.type
    with torch.autocast(device_type=tipo, dtype=torch.float16):
        mapa = predictor.classification_model.get_mh_distance_map(contraste)
    classe_px = mapa.min(dim=0)[1]

    saida = {k: [] for k in vazio}
    for flake in flakes:
        mask = flake.mask
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.astype(bool)).to(classe_px.device)
        else:
            mask = mask.to(classe_px.device).bool()

        if not bool(mask.any()):
            saida["frac_substrato"].append(1.0)
            for c in "bgr":
                saida[f"std_{c}"].append(0.0)
            continue

        saida["frac_substrato"].append(
            float((classe_px[mask] == 0).float().mean().item()))
        # O desvio precisa de pelo menos 2 pixels; com 1 o torch devolve NaN.
        pixels = contraste[mask]
        if pixels.shape[0] < 2:
            desvio = [0.0, 0.0, 0.0]
        else:
            desvio = torch.std(pixels, dim=0).float().tolist()
        for c, v in zip("bgr", desvio):
            saida[f"std_{c}"].append(float(v))

    return saida


@torch.inference_mode()
def calcular_frac_substrato(imagem_bgr, flakes):
    """Fracao de pixels de cada flake cuja classe MAIS PROXIMA e o substrato.

    Por que esta feature existe: o veto do MaskTerial mata um candidato quando a
    classe majoritaria dentro da mascara e 0 (fundo). Ha duas causas -- estar
    longe de tudo (o LIMIAR_MAHALANOBIS decide) ou estar PERTO do substrato.
    Esta grandeza isola a segunda, que limiar nenhum resgata, e foi a feature
    que separou o lixo 'vazio' no treino (mediana 0,68 contra 0,00 dos flakes).

    Custo: uma passada extra do AMM por imagem. O AMM e uma rede minuscula
    (3 entradas, 4 camadas) -- o caro no pipeline e o segmentador, entao isso
    quase nao aparece no tempo total.

    LIMITACAO CONHECIDA: no treino esta fracao foi medida sobre a mascara CRUA
    do segmentador; aqui so temos a mascara PREENCHIDA (o Flake ja vem com os
    buracos tapados). Buracos internos sao substrato visivel, entao o valor aqui
    tende a ser um pouco MAIOR. Como o coeficiente da feature e negativo, o erro
    empurra para 'lixo' -- conservador, nao otimista.
    """
    if not flakes:
        return []

    predictor = obter_predictor()
    imagem = torch.tensor(imagem_bgr, dtype=torch.uint8).to(predictor.device)
    contraste = predictor.calculate_contrast_image(imagem)

    tipo = predictor.device if isinstance(predictor.device, str) else predictor.device.type
    with torch.autocast(device_type=tipo, dtype=torch.float16):
        mapa = predictor.classification_model.get_mh_distance_map(contraste)

    # min sobre as classes: com qual delas cada pixel mais se parece.
    classe_px = mapa.min(dim=0)[1]

    fracoes = []
    for flake in flakes:
        mask = flake.mask
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.astype(bool)).to(classe_px.device)
        else:
            mask = mask.to(classe_px.device).bool()
        fracoes.append(float((classe_px[mask] == 0).float().mean().item())
                       if mask.any() else 1.0)
    return fracoes


def probabilidades_flake(imagem_bgr, flakes):
    """Probabilidade de cada flake ser flake de verdade, sem filtrar nada.

    Serve para quem precisa do ranking antes de decidir o corte -- por exemplo
    o testar_modelo_pasta.py no modo --orcamento, que olha a pasta inteira antes
    de escolher o limiar. Devolve None se o .joblib nao estiver presente.

    Anota cada flake com `.prob_flake` de passagem.
    """
    filtro = obter_filtro()
    if filtro is None or not flakes:
        return None

    valores = calcular_extras(imagem_bgr, flakes)
    valores.update({"c_b": [], "c_g": [], "c_r": []})
    for flake in flakes:
        c = np.asarray(flake.mean_contrast, dtype=float)
        valores["c_b"].append(c[0])
        valores["c_g"].append(c[1])
        valores["c_r"].append(c[2])

    # Derivadas: as MESMAS formulas do treinar_filtro.carregar(). Se mudarem la,
    # mudam aqui -- treino e inferencia calculando features diferentes e o tipo
    # de erro que nao levanta excecao nenhuma, so degrada o resultado em
    # silencio.
    for canal in "bgr":
        c = np.asarray(valores[f"c_{canal}"], dtype=float)
        s = np.asarray(valores[f"std_{canal}"], dtype=float)
        valores[f"snr_{canal}"] = np.abs(c) / np.maximum(s, 1e-4)

    # A ordem das colunas TEM de ser a mesma do treino; por isso o joblib
    # guarda a lista de nomes junto com o modelo.
    X = np.column_stack([valores[nome] for nome in filtro["features"]])
    probs = filtro["modelo"].predict_proba(X)[:, 1]
    for flake, p in zip(flakes, probs):
        flake.prob_flake = float(p)
    return probs


def filtrar_por_modelo(imagem_bgr, flakes, limiar=LIMIAR_PROB_FLAKE):
    """Mantem so os flakes que o modelo treinado considera flake de verdade.

    Substitui o filtro_por_cor, que era um TETO no contraste verde. O problema
    do teto: o contraste cresce em modulo com o numero de camadas, entao ele
    corta o que tem POUCO contraste (cola) e deixa passar o que tem MUITO
    (bulk) -- e o flake fino mora numa FAIXA, que nenhum teto isola.

    O modelo usa os tres canais, e a fronteira que ele aprendeu tem os
    coeficientes de azul e verde com SINAIS OPOSTOS (+3,7 e -3,3): o que decide
    nao e 'quao escuro', e a INCLINACAO do espectro. O flake fino tem contraste
    azul positivo (reflete mais azul que o substrato, por interferencia no
    SiO2); o bulk e grafite opaco, escuro nos tres canais. Medido: o bulk cai de
    78% para 5% dos falsos positivos.

    Cada flake sai anotado com `.prob_flake`, entao da para ordenar por ela
    (o mais promissor primeiro) mesmo depois de filtrar.

    Se o .joblib nao estiver presente, cai no filtro de cor antigo.
    """
    if not flakes:
        return []

    probs = probabilidades_flake(imagem_bgr, flakes)
    if probs is None:
        print("[aviso] modelo_filtro_flake.joblib nao encontrado -- "
              "usando o filtro de cor antigo (LIMIAR_CONTRASTE_G).")
        return filtrar_por_cor(flakes)

    aceitos = [f for f, p in zip(flakes, probs) if p >= limiar]
    aceitos.sort(key=lambda f: -f.prob_flake)
    return aceitos


def contornos_unidos(flakes, raio=RAIO_FUSAO_PX):
    """Une as máscaras de todos os flakes e devolve os contornos externos,
    fundindo fragmentos do mesmo flake.

    O modelo às vezes quebra um flake em pedaços (ou detecta a mesma região em
    segmentos sobrepostos). A gente soma todas as máscaras numa só, fecha vãos
    de até ~raio px (closing morfológico) e extrai os contornos: pedaços do
    mesmo flake viram um contorno; flakes de fato separados continuam separados.
    """
    if not flakes:
        return []

    altura, largura = flakes[0].mask.shape[:2]
    uniao = np.zeros((altura, largura), dtype=np.uint8)
    for flake in flakes:
        uniao[flake.mask.astype(np.uint8) > 0] = 255

    if raio > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * raio + 1, 2 * raio + 1))
        uniao = cv2.morphologyEx(uniao, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(uniao, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos


def desenhar_contornos(image, flakes, cor=(0, 0, 255),
                       espessura=ESPESSURA_CONTORNO, raio=RAIO_FUSAO_PX):
    """Desenha os contornos UNIDOS (fragmentos já fundidos) numa cópia da imagem.
    Sem bounding box nem rótulo -- pensado para o mosaico, onde importa só ver a
    forma e a posição do flake. Devolve a imagem marcada."""
    image = image.copy()
    cv2.drawContours(image, contornos_unidos(flakes, raio), -1, cor, espessura)
    return image


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEG_MODEL = "M2F"
CLS_MODEL = "AMM"


PP_MODEL = None
PP_MODEL_ROOT = None

_predictor = None


def obter_predictor():

    global _predictor

    if _predictor is None:
        segmentation_model, classification_model, postprocessing_model = load_models(
            seg_model_type=SEG_MODEL,
            seg_model_root=SEG_MODEL_ROOT,
            cls_model_type=CLS_MODEL,
            cls_model_root=CLS_MODEL_ROOT,
            pp_model_type=PP_MODEL,
            pp_model_root=PP_MODEL_ROOT,
            device=DEVICE,
        )
        assert hasattr(classification_model, "std_treshold"), (
            "AMM_head não tem mais 'std_treshold' -- o nome mudou na versão "
            "instalada do MaskTerial."
        )
        classification_model.std_treshold = LIMIAR_MAHALANOBIS
        _predictor = MaskTerial(
            segmentation_model=segmentation_model,
            classification_model=classification_model,
            postprocessing_model=postprocessing_model,
            score_threshold=SCORE_THRESHOLD,
            min_class_occupancy=MIN_CLASS_OCCUPANCY,
            size_threshold=SIZE_THRESHOLD,
            device=DEVICE,
        )

    return _predictor