import cv2
from maskterial import MaskTerial, load_models
from maskterial.structures import Flake
import torch
import matplotlib.pyplot as plt
import numpy as np
from caminhos import SEG_MODEL_ROOT, CLS_MODEL_ROOT
from parametros import SCORE_THRESHOLD, MIN_CLASS_OCCUPANCY, SIZE_THRESHOLD

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